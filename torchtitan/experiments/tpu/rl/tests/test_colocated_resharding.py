"""
Unit test for collocated vLLM + FSDP execution on TPU.

This script tests the end-to-end integration of torchtitan (FSDP) and vLLM on a single set of TPU devices.
Specifically, it performs the following:
1. Initializes a dummy vLLM engine alongside a torchtitan FSDP model.
2. Loads actual pre-trained weights into the FSDP model (via DCP or safetensors).
3. Evaluates the FSDP model's initial loss on a prompt to ensure weights were loaded correctly.
4. Synchronizes the parameters from the FSDP model directly into the vLLM worker's memory.
5. Performs text generation using vLLM to verify the weights were synced correctly and produce coherent English.
6. Evaluates the FSDP model's loss purely on the generated completion tokens to verify mathematical consistency.

To run this test:

```bash
source /home/jialeic_google_com/work/rl_poc/tpu_rl_env_2/bin/activate
sudo fuser -k /dev/vfio/* 2>/dev/null || true
export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=131008'
export JAX_MEM_FRACTION="0.5"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.5"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export VLLM_ENABLE_V1_MULTIPROCESSING="0"
export JAX_PLATFORMS="tpu"
export PYTHONPATH=$PYTHONPATH:.

# To test with DCP loading (default, tp=4, 1 replica):
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 torchtitan/experiments/tpu/rl/tests/test_colocated_resharding.py --vllm_tp 4 > minimal_dcp_repro_tp4.log 2>&1 

# To test with DCP loading with tp=2 (2 replicas):
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 torchtitan/experiments/tpu/rl/tests/test_colocated_resharding.py --vllm_tp 2 > minimal_dcp_repro_tp2.log 2>&1 

# To test with DCP loading with tp=1 (4 replicas):
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 torchtitan/experiments/tpu/rl/tests/test_colocated_resharding.py --vllm_tp 1 > minimal_dcp_repro_tp1.log 2>&1 

# To test with safetensors loading directly (disable DCP) with tp=2:
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 torchtitan/experiments/tpu/rl/tests/test_colocated_resharding.py --no-use_dcp --vllm_tp 2 > minimal_repro_tp2.log 2>&1 
```
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_PLATFORMS"] = "tpu"
import torch
from torchtitan.experiments.tpu.distributed_utils import maybe_init_distributed
maybe_init_distributed()
import torch_tpu
import torch.distributed as dist
import contextlib
import unittest.mock
import json
import pickle
import zmq

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine

from torchtitan.models.qwen3 import model_registry, Qwen3Model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    # We create a custom executor that broadcasts distributed commands to workers using ZMQ.
    # We do this because we want to run vLLM and FSDP natively in the same set of 
    # torchrun-managed processes without Ray overhead.
    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher
    class RayColocatedExecutor(ExecutorWithExternalLauncher):
        def _init_executor(self) -> None:
            super()._init_executor()
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            actual_rank = int(os.environ.get("GLOBAL_RANK", "0"))
            self.zmq_ctx = zmq.Context()
            self.cmd_sockets = []
            for i in range(1, tp_size):
                worker_rank = actual_rank + i
                sock = self.zmq_ctx.socket(zmq.PAIR)
                sock_path = f"ipc:///tmp/vllm-cmd-rank-{worker_rank}.sock"
                sock.connect(sock_path)
                self.cmd_sockets.append(sock)

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False, single_value=False):
            if kwargs is None:
                kwargs = {}
                
            if method == "initialize_from_config":
                kv_cache_configs = args[0]
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                if tp_size == 1:
                    tp_size = int(os.environ.get("VLLM_TP_SIZE", "4"))
                args = (kv_cache_configs * tp_size,)
                
            cmd = [method, args, kwargs]
            payload = pickle.dumps(cmd)
            rank = int(os.environ.get("RANK", "0"))
            print(f"@@@ [DRIVER Rank {rank}] ZMQ SEND: {method}")
            for sock in self.cmd_sockets:
                sock.send(payload)
            
            return super().collective_rpc(method, timeout, args, kwargs, non_block, single_value)

        def determine_available_memory(self) -> list[int]:
            rank = int(os.environ.get("RANK", "0"))
            print(f"@@@ [DRIVER Rank {rank}] ZMQ SEND: determine_available_memory")
            cmd = ["determine_available_memory", (), {}]
            payload = pickle.dumps(cmd)
            for sock in self.cmd_sockets:
                sock.send(payload)
            
            from vllm.v1.serial_utils import run_method
            print(f"@@@ [DRIVER Rank {rank}] Running determine_available_memory locally...")
            memory = run_method(self.driver_worker, "determine_available_memory", (), {})
            print(f"@@@ [DRIVER Rank {rank}] Running determine_available_memory locally DONE.")
            return [memory] * len(self.get_kv_cache_specs())
except ImportError:
    pass

@contextlib.contextmanager
def mock_vllm_distributed(tp_size: int):
    """
    Context manager that patches vLLM to support collocated multi-group TP on TPU.
    This effectively mocks out vLLM's internal distributed setup to rely on torchrun's
    environment and allow arbitrary tp_size <= world_size.
    """
    try:
        from tpu_inference.platforms.tpu_platform import TpuPlatform
        original_check = TpuPlatform.check_and_update_config
        def patched_check(cls, vllm_config):
            # Bypass single-host checks which strictly enforce tp_size == world_size
            with unittest.mock.patch.object(cls, '_prepare_singlehost_tpu_env'):
                original_check(vllm_config)
            if tp_size > 1:
                vllm_config.parallel_config.distributed_executor_backend = RayColocatedExecutor
        patch_tpu = unittest.mock.patch.object(TpuPlatform, 'check_and_update_config', new=classmethod(patched_check))
    except ImportError:
        patch_tpu = contextlib.nullcontext()

    try:
        # vLLM's TPUWorker explicitly requires the distributed_init_method to be a tcp:// URL.
        # If we don't patch this, it crashes instantly with: 
        # ValueError: Expected tcp://<host>:<port> distributed_init_method, got: 'env://'
        from tpu_inference.worker.tpu_worker import TPUWorker
        original_init_device = TPUWorker.init_device
        def patched_init_device(self):
            if self.distributed_init_method == "env://":
                host = os.environ.get('MASTER_ADDR', '127.0.0.1')
                port = os.environ.get('MASTER_PORT', '12345')
                self.distributed_init_method = f"tcp://{host}:{port}"
            original_init_device(self)
        patch_tpu_worker = unittest.mock.patch.object(TPUWorker, 'init_device', new=patched_init_device)
    except ImportError:
        patch_tpu_worker = contextlib.nullcontext()
        
    try:
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        import jax
        import numpy as np
        from jax.sharding import Mesh
        
        def patched_create_mesh_for_parallelism(self):
            # We must constrain vLLM's XLA mesh to the actual device assigned
            # to this global rank, rather than letting it assume it owns the whole 
            # host topology. Otherwise XLA collectives will deadlock between FSDP and vLLM.
            local_devices = list(jax.local_devices())
            global_rank = int(os.environ.get("GLOBAL_RANK", "0"))
            if len(local_devices) > global_rank:
                target_device = local_devices[global_rank]
            else:
                target_device = local_devices[0]
            mesh_devices = np.asarray([target_device]).reshape((1, 1))
            mesh = Mesh(mesh_devices, axis_names=("data", "model"))
            return mesh
        patch_tpu_runner = unittest.mock.patch.object(TPUModelRunner, '_create_mesh_for_parallelism', new=patched_create_mesh_for_parallelism)
        
    except ImportError:
        patch_tpu_runner = contextlib.nullcontext()
        
    try:
        from vllm.v1.executor.uniproc_executor import UniProcExecutor, ExecutorWithExternalLauncher
        original_distributed_args = UniProcExecutor._distributed_args
        
        def mock_distributed_args(self):
            # Crucial: we fake the rank variables vLLM sees to trick it into forming
            # distinct subgroup engines instead of one massive world engine.
            distributed_init_method, rank, local_rank = original_distributed_args(self)
            actual_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            actual_rank = int(os.environ.get("GLOBAL_RANK", "0"))
            tp_size = int(os.environ.get("VLLM_TP_SIZE", "4"))
            vllm_rank = actual_rank % tp_size
            vllm_local_rank = actual_local_rank % tp_size
            host = os.environ.get('MASTER_ADDR', '127.0.0.1')
            port = int(os.environ.get('MASTER_PORT', '12345'))
            return f"tcp://{host}:{port}", actual_rank, actual_local_rank

        patch_uni = unittest.mock.patch.object(UniProcExecutor, '_distributed_args', new=mock_distributed_args)
        patch_ext = unittest.mock.patch.object(ExecutorWithExternalLauncher, '_distributed_args', new=mock_distributed_args)
    except ImportError:
        patch_uni = contextlib.nullcontext()
        patch_ext = contextlib.nullcontext()

    @contextlib.contextmanager
    def debug_load_model():
        # Bypass vLLM's complex model loader which attempts to orchestrate
        # multi-host compilations and can trigger deadlocks when FSDP holds locks.
        try:
            from tpu_inference.runner.tpu_runner import TPUModelRunner
            original_load = TPUModelRunner.load_model
            def patched_load(self):
                self.device_config.device = torch.device("tpu")
                self.device = self.device_config.device
                self.vllm_config.device_config = self.device_config
                from vllm.model_executor.model_loader import get_model_loader
                model_loader = get_model_loader(self.load_config)
                from tpu_inference.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
                from vllm.config import set_current_vllm_config
                with set_vllm_model_wrapper_context(mesh=self.mesh), \
                     set_current_vllm_config(self.vllm_config):
                    model = model_loader.load_model(vllm_config=self.vllm_config, model_config=self.model_config)
                self.model = model
                self._initialize_attention_kernels()
            patcher = unittest.mock.patch.object(TPUModelRunner, 'load_model', new=patched_load)
            patcher.start()
            yield
            patcher.stop()
        except ImportError:
            yield

    with patch_uni, patch_ext, patch_tpu, patch_tpu_worker, patch_tpu_runner, debug_load_model():
        yield

@contextlib.contextmanager
def suppress_vllm_tpu_dummy_init_crash():
    """
    Context manager to suppress vLLM's initialization crashes when using dummy weights
    or when world_size != tp_size.
    """
    import unittest.mock
    try:
        import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils
    except ImportError:
        yield
        return

    def mock_init_dummy(param, low, high, seed):
        import torch
        with torch.no_grad():
            param.fill_(0)

    try:
        from torchtitan.distributed.parallel_dims import ParallelDims
        original_validate = ParallelDims._validate
        def patched_validate(self):
            # vLLM's wrapper logic tries to compute dp_shard based on a formula
            # that assumes the global world size is fully utilized by vLLM's mesh.
            # We override dp_shard here to allow arbitrary TP subgroups.
            product_without_dp = self.dp_replicate * self.cp * self.tp * self.pp
            if self.world_size % product_without_dp == 0:
                self.dp_shard = self.world_size // product_without_dp
            
            try:
                original_validate(self)
            except AssertionError as e:
                # Ignore the assertion since we intentionally mismatched dims for TP subgroups
                pass
        patch_dims = unittest.mock.patch.object(ParallelDims, '_validate', new=patched_validate)
    except ImportError:
        patch_dims = contextlib.nullcontext()

    try:
        import torch._ops
        # Allow internal ops into Torch compile graph so vLLM tracing doesn't crash
        torch.compiler.allow_in_graph(torch.ops._c10d_functional.all_reduce)
        torch.compiler.allow_in_graph(torch.ops._c10d_functional.wait_tensor)
        torch.compiler.allow_in_graph(torch.ops._c10d_functional.all_gather_into_tensor)
    except Exception as e:
        pass
        
    with unittest.mock.patch('vllm.model_executor.model_loader.weight_utils.initialize_single_dummy_weight', mock_init_dummy), patch_dims:
        yield

def get_prompt(prompt_rank):
    return f"""You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 → 1, 2
345 → 3, 4, 5
67 → 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28

What is the total digit sum of [{(prompt_rank+1)*10+prompt_rank}, 93, 53, 45]?"""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dcp", action=argparse.BooleanOptionalAction, default=True, help="Use DCP to load weights instead of safetensors directly")
    parser.add_argument("--model_path", type=str, default="/home/jialeic_google_com/work/rl_poc/torchtitan/assets/hf/Qwen3-0.6B", help="Path to the model")
    parser.add_argument("--vllm_tp", type=int, default=4, help="Tensor parallel size for vLLM")
    args, _ = parser.parse_known_args()

    rank = int(os.environ.get("RANK", "0"))
    os.environ["GLOBAL_RANK"] = str(rank)
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    dist.init_process_group("tpu_dist")
    
    tp_size = args.vllm_tp
    assert world_size % tp_size == 0, f"world_size {world_size} must be divisible by tp_size {tp_size}"
    
    # Patch vLLM's ParallelConfig check
    from vllm.config import ParallelConfig
    original_post_init = ParallelConfig.__post_init__
    def patched_post_init(self):
        original_post_init(self)
        self.world_size = world_size
    ParallelConfig.__post_init__ = patched_post_init
    
    print(f"@@@ [Rank {rank}] Initializing vLLM...")
    model_path = args.model_path

    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    
    from torchtitan.experiments.rl.models.vllm_registry import register_model_to_vllm_model_registry, VLLM_MODEL_NAME
    from torchtitan.config.configs import CompileConfig
    
    model_spec = model_registry("0.6B")
    register_model_to_vllm_model_registry(model_spec, compile_config=CompileConfig(enable=False))

    def get_config_dict():
        return {
            "architectures": [VLLM_MODEL_NAME],
            "model_type": "llama",
            "vocab_size": 151936,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
            "max_position_embeddings": 4096,
            "intermediate_size": 3072,
        }

    dummy_model_path = "/tmp/torchtitan-qwen3-0.6B-minimal"
    os.makedirs(dummy_model_path, exist_ok=True)
    config_path = os.path.join(dummy_model_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(get_config_dict(), f)

    os.environ["VLLM_TP_SIZE"] = str(tp_size)
    
    engine_args = EngineArgs(
        model=dummy_model_path,
        tokenizer=model_path,
        enforce_eager=False, 
        max_model_len=4096, 
        max_num_seqs=8,
        tensor_parallel_size=tp_size,
        load_format="dummy",  # will update weights later
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        gpu_memory_utilization=0.5,
        disable_log_stats=True
    )

    with mock_vllm_distributed(tp_size), suppress_vllm_tpu_dummy_init_crash():
        try:
            import torch._ops  # maybe those are not needed anymore, will need to double check
            torch.compiler.allow_in_graph(torch.ops._c10d_functional.all_reduce)
            torch.compiler.allow_in_graph(torch.ops._c10d_functional.wait_tensor)
            torch.compiler.allow_in_graph(torch.ops._c10d_functional.all_gather_into_tensor)
            torch.compiler.allow_in_graph(torch.ops._c10d_functional.reduce_scatter_tensor)
        except Exception as e:
            pass

        is_driver = (rank % tp_size == 0)
        vllm_rank = rank % tp_size

        if is_driver:
            print(f"@@@ [Rank {rank}] init LLMEngine...")
            engine = LLMEngine.from_engine_args(engine_args)
            print(f"@@@ [Rank {rank}] init LLMEngine DONE!")

            print(f"@@@ [Rank {rank}] Generating with dummy weights...")
            sampling_params = SamplingParams(temperature=1.0, max_tokens=100, ignore_eos=True)
            for i in range(rank, rank + tp_size):
                engine.add_request(f"dummy_{i}", {"prompt": get_prompt(i)}, sampling_params)
            outputs = []
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
            print(f"@@@ [Rank {rank}] Dummy generation done! Outputs: {outputs[0].outputs[0].text if outputs else ''}")

            if tp_size > 1:
                stop_payload = pickle.dumps(["stop_init", None, None])
                for sock in engine.model_executor.cmd_sockets:
                    sock.send(stop_payload)
        else:
            print(f"@@@ [Rank {rank}] init_worker...")
            vllm_config = engine_args.create_engine_config()
            from vllm.v1.worker.worker_base import WorkerWrapperBase
            worker_wrapper = WorkerWrapperBase(rpc_rank=vllm_rank)
            kwargs = dict(
                vllm_config=vllm_config,
                local_rank=rank % tp_size, # Assuming local rank should be 0 or 1. Actually wait, local_rank is just rank % tp_size.
                rank=rank,
                distributed_init_method="env://",
                is_driver_worker=False,
                shared_worker_lock=None,
            )
            all_kwargs = [{}] * tp_size
            all_kwargs[vllm_rank] = kwargs
            worker_wrapper.init_worker(all_kwargs=all_kwargs)
            worker_wrapper.init_device()
            worker_wrapper.load_model()
            from vllm.platforms import current_platform
            current_platform.update_block_size_for_backend(vllm_config)
        
            zmq_ctx = zmq.Context()
            zmq_sock = zmq_ctx.socket(zmq.PAIR)
            sock_path = f"ipc:///tmp/vllm-cmd-rank-{rank}.sock"
            if os.path.exists(sock_path):
                os.unlink(sock_path)
            zmq_sock.bind(sock_path)
            from vllm.v1.serial_utils import run_method
            print(f"@@@ [WORKER Rank {rank}] Listening on {sock_path}...")
            while True:
                payload = zmq_sock.recv()
                cmd = pickle.loads(payload)
                print(f"@@@ [WORKER Rank {rank}] ZMQ RECV: {cmd[0]}")
                if cmd[0] == "stop_init": break
                run_method(worker_wrapper, cmd[0], cmd[1], cmd[2])
                print(f"@@@ [WORKER Rank {rank}] ZMQ RECV DONE: {cmd[0]}")
            print(f"@@@ [Rank {rank}] init_worker DONE!")

        dist.barrier()

        # Ensure all local TPU ops from dummy generation are executed
        torch_tpu._internal.sync.synchronize(wait=True)

        # Everyone creates FSDP model
        print(f"@@@ [Rank {rank}] Creating Torchtitan FSDP Model...")
        model_args = model_spec.model
        with torch.device("meta"):
            model = Qwen3Model(model_args)
        
        from torch.distributed.fsdp import fully_shard
        model = model.to_empty(device="tpu")
        # model.freqs_cis = model.rope.cache.to("tpu")
        model.init_states(buffer_device=torch.device("tpu"))
        fully_shard(model)
        
        print(f"@@@ [Rank {rank}] Loading HF weights from {model_path}...")
        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
        from torch.distributed.tensor import DTensor, Replicate
        
        adapter = Qwen3StateDictAdapter(model_args, None)
        
        if args.use_dcp:
            print(f"@@@ [Rank {rank}] Using DCP to load weights...")
            import torch.distributed.checkpoint as dcp
            storage_reader = adapter.get_hf_storage_reader(model_path)
            hf_state_dict = adapter.to_hf(model.state_dict())
            dcp.load(hf_state_dict, storage_reader=storage_reader)
            torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        else:
            print(f"@@@ [Rank {rank}] Using safetensors to load weights directly...")
            from safetensors.torch import load_file
            hf_state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        
        model_state_dict = {k: v for k, v in model.state_dict().items()}
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                if isinstance(tensor, DTensor):
                    continue
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
                
        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=False),
        )
        
        print(f"@@@ [Rank {rank}] Evaluating FSDP model loss...")
        from transformers import AutoTokenizer
        import torch.nn.functional as F
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokens_cpu = tokenizer.encode(get_prompt(rank), return_tensors="pt")
        
        # HACK: FSDP evaluation involves global all_gather collectives for parameters and activations.
        # PyTorch XLA demands symmetric tensor shapes across all ranks for these collectives to work.
        # Since different ranks may have differently sized prompts, we pad them to a fixed shape (MAX_PROMPT_LEN)
        # using the ignore index (-100) so the extra tokens don't affect the cross_entropy loss.
        # TODO: Remove HACK when PyTorch XLA supports dynamic shape parameter sharding and evaluation.
        MAX_PROMPT_LEN = 256
        seq_len = tokens_cpu.shape[1]
        assert seq_len <= MAX_PROMPT_LEN, f"Sequence length {seq_len} exceeds max length {MAX_PROMPT_LEN}"
        
        padded_tokens = torch.zeros((1, MAX_PROMPT_LEN), dtype=torch.long)
        padded_labels = torch.full((1, MAX_PROMPT_LEN), -100, dtype=torch.long)
        
        padded_tokens[0, :seq_len] = tokens_cpu[0]
        padded_labels[0, :seq_len] = tokens_cpu[0]
        
        tokens = padded_tokens.to("tpu")
        labels = padded_labels.to("tpu")
        
        with torch.no_grad():
            logits = model(tokens)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            print(f"@@@ [Rank {rank}] FSDP Model Loss on prompt: {loss.item()}")
            assert loss.item() < 2.5, f"Expected loss < 2.5, got {loss.item()}"
        
        print(f"@@@ [Rank {rank}] Syncing weights FSDP -> vLLM...")
        
        if is_driver:
            vllm_model = engine.model_executor.driver_worker.get_model()
        else:
            vllm_model = worker_wrapper.worker.model_runner.model
            
        target_sd = dict(vllm_model.named_parameters())
        
        def get_local_vllm_tensor(source_p, target_t_dtensor):
            val = source_p
            if hasattr(val, "full_tensor"):
                full_val = val.full_tensor()
            else:
                full_val = val
                
            if hasattr(target_t_dtensor, "device_mesh") and hasattr(target_t_dtensor, "placements"):
                from torch.distributed.tensor import DTensor, Replicate
                val_dtensor = DTensor.from_local(
                    full_val.to(target_t_dtensor.device),
                    device_mesh=target_t_dtensor.device_mesh,
                    placements=[Replicate()] * target_t_dtensor.device_mesh.ndim,
                )
                val_dtensor_sharded = val_dtensor.redistribute(
                    device_mesh=target_t_dtensor.device_mesh,
                    placements=target_t_dtensor.placements,
                )
                return val_dtensor_sharded.to_local()
            else:
                if hasattr(target_t_dtensor, "to_local"):
                    target_local = target_t_dtensor.to_local()
                else:
                    target_local = target_t_dtensor
                    
                if target_local.shape == full_val.shape:
                    return full_val
                    
                for dim in range(len(target_local.shape)):
                    if target_local.shape[dim] != full_val.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = (rank % tp_size) * shard_size
                        indices = [slice(None)] * len(target_local.shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        return full_val[tuple(indices)]
                return full_val

        print(f"@@@ [Rank {rank}] Calling FSDP.summon_full_params...")
        # TODO: confirm this is all through fast ICI
        with FSDP.summon_full_params(model, recurse=True, writeback=False):
            source_sd = dict(model.named_parameters())
            
            # If weight tying is enabled, named_parameters() might only return tok_embeddings.weight.
            # We explicitly add lm_head.weight so it gets synced to vLLM.
            tok_key = next((k for k in source_sd.keys() if "tok_embeddings.weight" in k), None)
            lm_key = next((k for k in source_sd.keys() if "lm_head.weight" in k), None)
            if tok_key and not lm_key:
                new_lm_key = tok_key.replace("tok_embeddings", "lm_head")
                source_sd[new_lm_key] = source_sd[tok_key]
                
            synced_keys = []
            with torch.no_grad():
                for name, source_p in source_sd.items():
                    print(f"@@@ [Rank {rank}] Syncing param {name}...")
                    clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("module.", "")
                    target_name = f"model.{clean_name}"
                    
                    if target_name in target_sd:
                        target_t_dtensor = target_sd[target_name]
                        target_t = target_t_dtensor.to_local() if hasattr(target_t_dtensor, "to_local") else target_t_dtensor
                        val = get_local_vllm_tensor(source_p, target_t_dtensor)
                        
                        if target_t.shape == val.shape:
                            target_t.copy_(val)
                            synced_keys.append(target_name)
                        else:
                            if rank == 0: print(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
                    else:
                        if clean_name in target_sd:
                            target_name = clean_name
                            target_t_dtensor = target_sd[target_name]
                            target_t = target_t_dtensor.to_local() if hasattr(target_t_dtensor, "to_local") else target_t_dtensor
                            
                            val = get_local_vllm_tensor(source_p, target_t_dtensor)
                            if target_t.shape == val.shape:
                                target_t.copy_(val)
                                synced_keys.append(target_name)
                            else:
                                if rank == 0: print(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
                    
                    source_sd[name] = None
                    
            print(f"@@@ [Rank {rank}] Synced {len(synced_keys)} keys to vLLM.")
        
        torch_tpu._internal.sync.synchronize(wait=True)

        if is_driver:
            print(f"@@@ [Rank {rank}] Starting final generation...")
            sampling_params = SamplingParams(temperature=1.0, max_tokens=100, ignore_eos=True)            
            for i in range(rank, rank + tp_size):
                engine.add_request(f"final_{i}", {"prompt": get_prompt(i)}, sampling_params)
            outputs = []
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
                        
            # Sort outputs by request ID to ensure matching ranks
            outputs = sorted(outputs, key=lambda x: int(x.request_id.split('_')[1]))
            
            if tp_size > 1:
                stop_payload = pickle.dumps(["stop", None, None])
                for sock in engine.model_executor.cmd_sockets:
                    sock.send(stop_payload)
                    
            print(f"@@@ [Rank {rank}] Final generation done! Outputs: {outputs[0].outputs[0].text if outputs else ''}")

        else:
            print(f"@@@ [Rank {rank}] Entering generate loop...")
            while True:
                payload = zmq_sock.recv()
                cmd = pickle.loads(payload)
                if cmd[0] == "stop": break
                run_method(worker_wrapper, cmd[0], cmd[1], cmd[2])
            print(f"@@@ [Rank {rank}] Worker generation done.")
            
        # HACK: We leverage ICI via a global collective to sync the generated text.
        # We cannot use `dist.broadcast_object_list` because it combines async ZeroMQ communication
        # with TPU collectives, which leads to stream desynchronizations and AtenToCopy hangs.
        # Furthermore, PyTorch XLA demands symmetric tensor shapes across all ranks for collectives.
        # Therefore, we statically pad the strings into a fixed-size MAX_LEN tensor.
        # TODO: Remove HACK when XLA supports dynamic shape collectives or safe broadcast_object_list.
        # Normal usage should be `dist.broadcast_object_list(text_list, src=driver_rank)`.
        MAX_LEN = 4096
        cpu_local_tensor = torch.zeros(world_size, MAX_LEN, dtype=torch.int32)
        
        if is_driver:
            for idx, i in enumerate(range(rank, rank + tp_size)):
                gen_text = outputs[idx].outputs[0].text if outputs else ''
                bytes_text = gen_text.encode('utf-8')
                assert len(bytes_text) < MAX_LEN
                if len(bytes_text) > 0:
                    cpu_local_tensor[i, :len(bytes_text)] = torch.tensor(list(bytes_text), dtype=torch.int32)
        
        # Ensure all TPU ops from generation are finished to prevent stream desynchronization
        # before we execute the ICI collective.
        # HACK: vLLM generation launches PyTorch XLA kernels asynchronously. Since different ranks
        # may generate different lengths of text (ignore_eos=False), their Python loops finish at
        # different times. If a rank proceeds to a global TPU collective (like FSDP or all_reduce) 
        # while another rank is still launching local vLLM kernels, it will cause an AtenToCopy deadlock.
        # We must explicitly synchronize the TPU streams here.
        # TODO: Remove HACK when vLLM natively supports safe transitioning to global XLA collectives.
        # Normal usage should just rely on standard dist collectives without manual stream syncs.
        torch_tpu._internal.sync.synchronize(wait=True)
        
        # HACK: Force a CPU-level barrier so no rank queues the next TPU collective until ALL ranks
        # have fully finished their generation and stream synchronization.
        # TODO: Remove HACK once PyTorch XLA handles desynchronized collective launches gracefully.
        dist.barrier()
        
        local_tensor = cpu_local_tensor.to("tpu")
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        
        my_tensor = local_tensor[rank].cpu().tolist()
        if 0 in my_tensor:
            my_tensor = my_tensor[:my_tensor.index(0)]
        generated_text = bytes(my_tensor).decode('utf-8')

        assert len(generated_text) > 0, "Generated completion is empty"
        
        my_prompt = get_prompt(rank)
        final_prompt = my_prompt + generated_text
        
        print(f"@@@ [Rank {rank}] Evaluating FSDP model loss on final completion...")
        tokens_completion_cpu = tokenizer.encode(final_prompt, return_tensors="pt")
        prompt_tokens_cpu = tokenizer.encode(my_prompt, return_tensors="pt")
        prompt_len = prompt_tokens_cpu.shape[1]
        seq_len = tokens_completion_cpu.shape[1]
        
        # HACK: Similarly, we must pad the completions to a fixed shape for FSDP evaluation.
        # Since ignore_eos=False during generation, different ranks generated different lengths
        # of sequences, which will crash FSDP's strict shape requirements.
        MAX_SEQ_LEN = 512
        assert seq_len <= MAX_SEQ_LEN, f"Sequence length {seq_len} exceeds max length {MAX_SEQ_LEN}"
        
        padded_tokens = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.long)
        padded_labels = torch.full((1, MAX_SEQ_LEN), -100, dtype=torch.long)
        
        padded_tokens[0, :seq_len] = tokens_completion_cpu[0]
        padded_labels[0, :seq_len] = tokens_completion_cpu[0]
        padded_labels[0, :prompt_len] = -100
        
        tokens_completion = padded_tokens.to("tpu")
        labels_completion = padded_labels.to("tpu")
        
        torch_tpu._internal.sync.synchronize(wait=True)
        
        with torch.no_grad():
            logits = model(tokens_completion)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_completion[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            print(f"@@@ [Rank {rank}] FSDP Model Loss on completion: {loss.item()}")
            assert loss.item() < 2.5, f"Expected completion loss < 2.5, got {loss.item()}"
            
    print(f"@@@ [Rank {rank}] Test completed successfully!")
    
    # Avoid torchtpu teardown segfaults during Python garbage collection
    dist.barrier()
    os._exit(0)

if __name__ == "__main__":
    main()
