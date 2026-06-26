import contextlib
import json
import logging
import os
import pickle
import traceback
import unittest.mock

import torch
from torch.distributed import fsdp

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"
ZMQ_BASE_PORT = 9500
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "12345"

try:
    import zmq
    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher
    from vllm.v1.serial_utils import run_method
    
    class RayColocatedExecutor(ExecutorWithExternalLauncher):
        """
        A custom vLLM executor that broadcasts distributed commands to workers using ZeroMQ (ZMQ).
        
        Why this is needed:
        By default, vLLM uses Ray or Python multiprocessing to manage distributed execution. However, 
        in our setup, the processes are already managed by `torchrun` (to support PyTorch FSDP natively).
        If vLLM attempts to spawn its own Ray cluster or processes, it will clash with the existing 
        TPU resource allocations and cause deadlocks. This executor bypasses vLLM's process management 
        by using ZMQ to send generation commands to existing torchrun worker processes.
        """
        def _init_executor(self) -> None:
            super()._init_executor()
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            actual_rank = int(os.environ.get("RANK", "0"))
            self.zmq_ctx = zmq.Context()
            self.cmd_sockets = []
            
            # Connect to the ZMQ sockets of the worker ranks within our tensor parallel group
            for i in range(1, tp_size):
                worker_rank = actual_rank + i
                sock = self.zmq_ctx.socket(zmq.PAIR)
                
                addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "").split(",")
                if len(addresses) > worker_rank and addresses[worker_rank]:
                    host = addresses[worker_rank].split(":")[0]
                    my_host = addresses[actual_rank].split(":")[0]
                    if host == my_host:
                        host = DEFAULT_MASTER_ADDR
                    port = ZMQ_BASE_PORT + worker_rank
                    sock_path = f"tcp://{host}:{port}"
                else:
                    sock_path = f"ipc:///tmp/vllm-cmd-rank-{worker_rank}.sock"
                sock.connect(sock_path)
                self.cmd_sockets.append(sock)

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False, single_value=False):
            if kwargs is None:
                kwargs = {}
                
            # Intercept config initialization to ensure correct TP scaling
            if method == "initialize_from_config":
                kv_cache_configs = args[0]
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                if tp_size == 1:
                    tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
                args = (kv_cache_configs * tp_size,)
                
            # Serialize the RPC command and broadcast it over ZMQ to all workers
            cmd = [method, args, kwargs]
            payload = pickle.dumps(cmd)
            for sock in self.cmd_sockets:
                sock.send(payload)
            
            return super().collective_rpc(method, timeout, args, kwargs, non_block, single_value)

        def determine_available_memory(self) -> list[int]:
            # Instruct workers to profile their memory over ZMQ
            cmd = ["determine_available_memory", (), {}]
            payload = pickle.dumps(cmd)
            for sock in self.cmd_sockets:
                sock.send(payload)
            
            memory = run_method(self.driver_worker, "determine_available_memory", (), {})
            return [memory] * len(self.get_kv_cache_specs())
except ImportError:
    pass

@contextlib.contextmanager
def mock_vllm_distributed(tp_size: int):
    """
    Context manager that patches vLLM internals to support collocated multi-group TP on TPU.
    
    This effectively mocks out vLLM's internal distributed checks and mesh creations to rely 
    on the torchrun environment and allow arbitrary tp_size <= world_size.
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
                host = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
                port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
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
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if len(local_devices) > local_rank:
                target_device = local_devices[local_rank]
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
            actual_rank = int(os.environ.get("RANK", "0"))
            host = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
            port = int(os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT))
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

    When `load_format="dummy"` is used, vLLM allocates dummy tensors and
    internally calls `torch._sync(param)` on them. On PyTorch TPU, calling
    `torch._sync()` on these newly created dummy functional tensors triggers
    an internal C++ assertion (`isFunctionalTensor(t) INTERNAL ASSERT FAILED`).

    This mock allows vLLM to allocate the dummy memory buffers it needs for
    profiling and KV cache allocation safely. Once the vLLM engine finishes 
    starting up, our `sync_weights()` method is called to overwrite these dummy 
    buffers with the real initialized weights from the TorchTitan FSDP model.
    """
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
        torch.compiler.allow_in_graph(torch.ops._c10d_functional.reduce_scatter_tensor)
    except Exception as e:
        pass

    with unittest.mock.patch('vllm.model_executor.model_loader.weight_utils.initialize_single_dummy_weight', mock_init_dummy), patch_dims:
        yield


class VLLMSampler:
    """
    Handles sampling (generation) using vLLM for GRPO training on TPU.
    
    This class wraps the vLLM engine, dynamically configuring it to match the
    TorchTitan model architecture. During the training step, it pulls the FSDP 
    weights into its internal cache to generate rollouts, avoiding the overhead 
    of loading real weights from disk during initialization. It supports running
    multiple independent vLLM engines (e.g. TP=1 or TP=2) in a unified environment.
    """
    def __init__(self, job_config):
        self.job_config = job_config
        self.vllm_engine = None
        self._init_vllm()

    @property
    def rank(self) -> int:
        return int(os.environ.get("RANK", "0"))

    @property
    def tp_size(self) -> int:
        return getattr(self.job_config.sampler, "vllm_tensor_parallel_size", 1)
        
    @property
    def is_driver(self) -> bool:
        """Indicates if this process rank is the driver for its TP subgroup."""
        return self.rank % self.tp_size == 0

    def _get_config_dict_from_vllm_registry(self):
        """
        Extracts the actual configuration dimensions to allow for accurate KV cache allocation.
        """
        from torchtitan.experiments.rl.models.vllm_registry import VLLM_MODEL_NAME
        model_args = getattr(self.job_config.model_spec, "model", None) if hasattr(self.job_config, 'model_spec') else None
        layers = getattr(model_args, "layers", []) if model_args else []
        first_layer_attn = getattr(layers[0], "attention", None) if isinstance(layers, list) and layers else None

        return {
            "architectures": [VLLM_MODEL_NAME],
            "model_type": "llama",
            "vocab_size": getattr(model_args, "vocab_size", 256000) if model_args else 256000,
            "num_hidden_layers": len(layers) if isinstance(layers, list) and layers else getattr(model_args, "n_layers", 4) if model_args else 4,
            "num_attention_heads": getattr(first_layer_attn, "n_heads", 16) if first_layer_attn else 16,
            "num_key_value_heads": getattr(first_layer_attn, "n_kv_heads", 2) if first_layer_attn else 2,
            "hidden_size": getattr(model_args, "dim", 1024) if model_args else 1024,
            "max_position_embeddings": getattr(getattr(model_args, "rope", None), "max_seq_len", 1024) if model_args else 1024
        }

    def _get_tokenizer_name(self):
        """
        Determines the tokenizer name or path from the job configuration.
        """
        if hasattr(self.job_config, 'checkpoint') and self.job_config.checkpoint.initial_load_path:
            return self.job_config.checkpoint.initial_load_path
            
        if hasattr(self.job_config, 'hf_assets_path') and self.job_config.hf_assets_path:
            return self.job_config.hf_assets_path
            
        return DEFAULT_TOKENIZER_NAME

    def _create_dummy_model_config(self):
        """
        Creates a dummy config.json file on disk for vLLM to read dimensions from.
        """
        vllm_model_str = f"torchtitan-{self.job_config.model.name}-{self.job_config.model.flavor}"
        dummy_model_path = f"/tmp/{vllm_model_str}"
        os.makedirs(dummy_model_path, exist_ok=True)
        config_path = os.path.join(dummy_model_path, "config.json")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(self._get_config_dict_from_vllm_registry(), f)
        return dummy_model_path

    def _send_zmq_cmd(self, method, args=None, kwargs=None):
        """
        Sends a command via ZMQ to all other ranks' worker processes in the vLLM engine.
        """
        if self.vllm_engine is None:
            return
        payload = pickle.dumps([method, args, kwargs])
        engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
        executor = getattr(engine, "model_executor", None)
        for sock in getattr(executor, "cmd_sockets", []):
            sock.send(payload)

    def _init_zmq_socket(self):
        """
        Initializes a ZeroMQ PAIR socket for the current worker rank to receive commands from the driver.
        """
        zmq_ctx = zmq.Context()
        self.zmq_sock = zmq_ctx.socket(zmq.PAIR)

        addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "").split(",")
        if len(addresses) > self.rank and addresses[self.rank]:
            port = ZMQ_BASE_PORT + self.rank
            sock_path = f"tcp://0.0.0.0:{port}"
            self.zmq_sock.bind(sock_path)
        else:
            sock_path = f"ipc:///tmp/vllm-cmd-rank-{self.rank}.sock"
            if os.path.exists(sock_path):
                os.unlink(sock_path)
            self.zmq_sock.bind(sock_path)

    def _worker_loop_zmq(self, stop_cmd):
        """
        Continuously polls the ZMQ socket for commands from the driver and executes them until a stop command is received.
        """
        from vllm.v1.serial_utils import run_method
        while True:
            payload = self.zmq_sock.recv()
            cmd = pickle.loads(payload)
            if cmd[0] == stop_cmd:
                break
            run_method(self.worker_wrapper, cmd[0], cmd[1], cmd[2])

    def _init_vllm(self):
        """
        Initializes the vLLM engine alongside the FSDP training process.

        If TP=1, each rank runs an independent, local vLLM engine.
        If TP>1, it initializes distributed LLM engines utilizing the patched RayColocatedExecutor
        to run within the existing torchrun/FSDP process group without using Ray.
        Ranks where (rank % TP == 0) become drivers for their subgroups, while others act as 
        ZMQ workers awaiting commands.
        """
        logger.info(f"Initializing vLLM engine (Data Parallel mode, TP={self.tp_size})...")
        
        os.environ['SKIP_JAX_PRECOMPILE'] = '1'
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        os.environ["VLLM_TP_SIZE"] = str(self.tp_size)

        try:
            from torchtitan.experiments.rl.models.vllm_registry import register_model_to_vllm_model_registry
            if hasattr(self.job_config, 'model_spec'):
                compile_config = getattr(self.job_config, 'compile', None)
                register_model_to_vllm_model_registry(self.job_config.model_spec, compile_config=compile_config)
        except Exception as e:
            logger.warning(f"Could not register model to vLLM registry: {e}")

        dummy_model_path = self._create_dummy_model_config()
        tokenizer_name = self._get_tokenizer_name()

        # Patch vLLM's ParallelConfig check so it allows tp_size < world_size
        from vllm.config import ParallelConfig
        original_post_init = ParallelConfig.__post_init__
        def patched_post_init(self_config):
            original_post_init(self_config)
            actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self_config.world_size = actual_world_size
            self_config.data_parallel_size = actual_world_size // (self_config.tensor_parallel_size * getattr(self_config, "pipeline_parallel_size", 1))
        ParallelConfig.__post_init__ = patched_post_init

        try:
            from vllm.config import SchedulerConfig
            original_get_scheduler_cls = SchedulerConfig.get_scheduler_cls
            def patched_get_scheduler_cls(self):
                cls = original_get_scheduler_cls(self)
                if cls.__name__ == 'DPScheduler':
                    import vllm.v1.core.sched.scheduler as vllm_scheduler
                    return vllm_scheduler.Scheduler
                return cls
            SchedulerConfig.get_scheduler_cls = patched_get_scheduler_cls
        except ImportError:
            pass

        try:
            from vllm.engine.arg_utils import EngineArgs
            from torchtitan.experiments.rl.models.vllm_registry import VLLM_MODEL_NAME
            
            engine_args = EngineArgs(
                model=dummy_model_path,
                tokenizer=tokenizer_name,
                enforce_eager=self.job_config.sampler.vllm_enforce_eager, 
                max_model_len=self.job_config.sampler.vllm_max_model_len, 
                max_num_seqs=8,
                tensor_parallel_size=self.tp_size,
                load_format="dummy",
                hf_overrides={"architectures": [VLLM_MODEL_NAME]},
                gpu_memory_utilization=self.job_config.sampler.vllm_gpu_memory_utilization,
                disable_log_stats=True
            )

            with mock_vllm_distributed(self.tp_size), suppress_vllm_tpu_dummy_init_crash():
                vllm_rank = self.rank % self.tp_size

                if self.is_driver:
                    from vllm.engine.llm_engine import LLMEngine
                    self.vllm_engine = LLMEngine.from_engine_args(engine_args)
                    self.worker_wrapper = None
                    if self.tp_size > 1:
                        self._send_zmq_cmd("stop_init")
                else:
                    self.vllm_engine = None
                    vllm_config = engine_args.create_engine_config()
                    from vllm.v1.worker.worker_base import WorkerWrapperBase
                    self.worker_wrapper = WorkerWrapperBase(rpc_rank=vllm_rank)
                    kwargs = dict(
                        vllm_config=vllm_config,
                        local_rank=self.rank % self.tp_size,
                        rank=self.rank,
                        distributed_init_method="env://",
                        is_driver_worker=False,
                        shared_worker_lock=None,
                    )
                    all_kwargs = [{}] * self.tp_size
                    all_kwargs[vllm_rank] = kwargs
                    self.worker_wrapper.init_worker(all_kwargs=all_kwargs)
                    self.worker_wrapper.init_device()
                    self.worker_wrapper.load_model()
                    
                    from vllm.platforms import current_platform
                    current_platform.update_block_size_for_backend(vllm_config)
                
                    self._init_zmq_socket()
                    self._worker_loop_zmq("stop_init")

            logger.info("vLLM engine initialized successfully on rank %d.", self.rank)
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.error(traceback.format_exc())
            self.vllm_engine = None
            self.worker_wrapper = None
            
        finally:
            ParallelConfig.__post_init__ = original_post_init

    @staticmethod
    def _get_local_vllm_tensor(source_p, target_t_dtensor, rank, tp_size):
        """
        Converts and reshards a PyTorch/FSDP parameter tensor into the expected local shape 
        and placement required by vLLM for the current worker rank.

        Args:
            source_p (torch.Tensor): The source parameter from the PyTorch FSDP model.
            target_t_dtensor (torch.Tensor or DTensor): The target parameter structure from vLLM.
            rank (int): The current worker rank for calculating shard offsets.
            tp_size (int): Tensor parallelism dimension for vllm

        Returns:
            torch.Tensor: The appropriately sliced local tensor ready to be copied into vLLM's memory.
        """
        source_local = source_p.to_local() if hasattr(source_p, "to_local") else source_p
        target_local = target_t_dtensor.to_local() if hasattr(target_t_dtensor, "to_local") else target_t_dtensor
        
        if target_local.shape == source_local.shape:
            return source_local
            
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

    @staticmethod
    def _clean_module_name(name: str) -> str:
        """Helper method to remove wrapper prefixes from module names."""
        return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("module.", "")

    def sync_weights(self, source_model: torch.nn.Module = None, state_dict: dict = None):
        """
        Synchronizes the parameters from the active FSDP TorchTitan model directly into 
        the collocated vLLM engine's memory.

        This avoids the need to save weights to disk and load them back into vLLM. 
        It operates by calling `FSDP.summon_full_params` (or similar mechanisms depending on the setup)
        to gather the distributed weights across the fast Inter-Core Interconnect (ICI) network on the TPU. 
        Once the full weights or relevant shards are available locally, they are copied block-by-block 
        into vLLM's corresponding tensor buffers.

        Args:
            source_model (torch.nn.Module): The TorchTitan FSDP model to pull weights from.
            state_dict (dict): Optional pre-fetched state dict to load from directly.
        """
        if self.vllm_engine is None and not hasattr(self, "worker_wrapper"):
            return
            
        logger.info("Syncing weights to vLLM engine on rank %d...", self.rank)
        
        if self.is_driver:
            engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
            vllm_model = engine.model_executor.driver_worker.get_model()
        else:
            vllm_model = self.worker_wrapper.worker.model_runner.model
            
        target_model = getattr(vllm_model, "model", vllm_model)
        target_sd = target_model.state_dict()

        synced_keys = []
        if state_dict is not None:
            source_sd = state_dict
        elif source_model is not None:
            source_sd = source_model.state_dict()
        else:
            raise ValueError("Must provide either source_model or state_dict")
        
        tok_key = next((k for k in source_sd.keys() if "tok_embeddings.weight" in k), None)
        lm_key = next((k for k in source_sd.keys() if "lm_head.weight" in k), None)
        if tok_key and not lm_key:
            new_lm_key = tok_key.replace("tok_embeddings", "lm_head")
            source_sd[new_lm_key] = source_sd[tok_key]

        with torch.no_grad():
            for name, source_p in source_sd.items():
                clean_name = self._clean_module_name(name)
                target_name = clean_name
                if target_name not in target_sd:
                    target_name = f"model.{clean_name}"
                    
                if target_name in target_sd:
                    target_t_dtensor = target_sd[target_name]
                    target_t = target_t_dtensor.to_local() if hasattr(target_t_dtensor, "to_local") else target_t_dtensor
                    val = self._get_local_vllm_tensor(source_p, target_t_dtensor, self.rank, self.tp_size)
                    
                    if list(target_t.shape) == list(val.shape):
                        target_t.copy_(val)
                        synced_keys.append(target_name)
                        if "tok_embeddings" in target_name:
                            if hasattr(target_model, "lm_head") and hasattr(target_model.lm_head, "weight"):
                                lm_weight = target_model.lm_head.weight
                                if hasattr(lm_weight, "to_local"):
                                    lm_weight = lm_weight.to_local()
                                lm_weight.copy_(val)
                                if self.rank == 0: logger.info("Explicitly copied tok_embeddings to lm_head.weight!")
                    else:
                        if self.rank == 0: logger.warning(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
                else:
                    if self.rank == 0: logger.warning(f"Key missing in target_sd: {target_name}")
                
                source_sd[name] = None
            
        if self.rank == 0: logger.info(f"Successfully synced {len(synced_keys)} parameters to vLLM.")
        
        import torch_tpu
        torch_tpu._internal.sync.synchronize(wait=True)

    def generate(self, prompt_ids_repeated: torch.Tensor, sampling_params) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates completions for a batch of prompts using vLLM engine, for both colocated and non-colocated cases.

        If TP > 1, the driver rank controls the generation loop while other ranks execute the worker 
        loop via ZMQ commands. Once generation completes, the driver rank broadcasts the generated 
        tokens and log probabilities over TPU ICI (fast hardware interconnect) to all other ranks in 
        its TP subgroup so they remain in sync for the PPO/GRPO update.

        Args:
            prompt_ids_repeated (torch.Tensor): A tensor containing the tokenized prompt IDs.
            sampling_params (SamplingParams): vLLM sampling configurations (temperature, max_tokens, etc.).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The concatenated prompt + generated token IDs.
                - The log probabilities of the generated tokens.
        """
        if self.is_driver and self.vllm_engine is None:
            raise RuntimeError("vLLM engine is not initialized.")
            
        sampling_params.logprobs = 1
        
        local_prompts_list = prompt_ids_repeated.cpu().tolist()
        
        if self.tp_size > 1:
            use_zmq_broadcast = int(os.environ.get("VLLM_USE_ZMQ_BROADCAST", "1")) == 1
            if use_zmq_broadcast:
                if self.is_driver:
                    import pickle
                    engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
                    executor = getattr(engine, "model_executor", None)
                    sockets = getattr(executor, "cmd_sockets", [])
                    
                    global_prompt_ids_list = list(local_prompts_list)
                    for sock in sockets:
                        payload = sock.recv()
                        worker_prompts = pickle.loads(payload)
                        global_prompt_ids_list.extend(worker_prompts)
                else:
                    import pickle
                    payload = pickle.dumps(local_prompts_list)
                    self.zmq_sock.send(payload)
                    global_prompt_ids_list = None
            else:
                import torch.distributed as dist
                world_size = dist.get_world_size()
                
                if not hasattr(self, "tp_group"):
                    for i in range(0, world_size, self.tp_size):
                        ranks = list(range(i, i + self.tp_size))
                        group = dist.new_group(ranks)
                        if self.rank in ranks:
                            self.tp_group = group
                
                local_prompts = prompt_ids_repeated.clone()
                group_prompts = [torch.zeros_like(local_prompts) for _ in range(self.tp_size)]
                dist.all_gather(group_prompts, local_prompts, group=self.tp_group)
                all_prompt_ids = torch.cat(group_prompts, dim=0)
                global_prompt_ids_list = all_prompt_ids.cpu().tolist()
        else:
            global_prompt_ids_list = local_prompts_list
        
        completions = []
        token_log_probs_list = []

        if self.is_driver:
            prompts = []
            for p in global_prompt_ids_list:
                try:
                    eos_idx = p.index(151643)
                    start_idx = eos_idx + 1
                except ValueError:
                    try:
                        eos_idx = p.index(0)
                        start_idx = eos_idx + 1
                    except ValueError:
                        start_idx = 0
                        
                if start_idx >= len(p):
                    start_idx = len(p) - 1
                
                prompts.append({"prompt_token_ids": p[start_idx:]})
                
            for idx, p in enumerate(prompts):
                self.vllm_engine.add_request(str(idx), p, sampling_params)

            outputs = []
            while self.vllm_engine.has_unfinished_requests():
                step_outputs = self.vllm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
            
            outputs.sort(key=lambda x: int(x.request_id))

            if self.tp_size > 1:
                self._send_zmq_cmd("stop")
                
            max_new_tokens = sampling_params.max_tokens
            for output in outputs:
                completion_output = output.outputs[0]
                comp_ids = list(completion_output.token_ids)
                
                if len(comp_ids) < max_new_tokens:
                    comp_ids.extend([0] * (max_new_tokens - len(comp_ids)))
                elif len(comp_ids) > max_new_tokens:
                    comp_ids = comp_ids[:max_new_tokens]
                    
                if self.is_driver and output == outputs[0]:
                    logger.debug(f"vLLM Generated tokens length: {len(completion_output.token_ids)}")
                    
                completions.append(comp_ids)
                
                step_logprobs = []
                for i, token_id in enumerate(completion_output.token_ids):
                    pos_logprobs = completion_output.logprobs[i]
                    if token_id in pos_logprobs:
                        step_logprobs.append(pos_logprobs[token_id].logprob)
                    else:
                        step_logprobs.append(0.0)
                        
                if len(step_logprobs) < max_new_tokens:
                    step_logprobs.extend([0.0] * (max_new_tokens - len(step_logprobs)))
                elif len(step_logprobs) > max_new_tokens:
                    step_logprobs = step_logprobs[:max_new_tokens]
                    
                token_log_probs_list.append(step_logprobs)
        elif getattr(self, "worker_wrapper", None) is not None:
            self._worker_loop_zmq("stop")
                    
        if self.tp_size > 1:
            use_zmq_broadcast = int(os.environ.get("VLLM_USE_ZMQ_BROADCAST", "1")) == 1
            local_batch_size = len(prompt_ids_repeated)
            if use_zmq_broadcast:
                # Use ZMQ for broadcasting. dist.broadcast does not work for multihost cases
                # because the XLA collectives will deadlock between FSDP and vLLM.
                if self.is_driver:
                    import pickle
                    engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
                    executor = getattr(engine, "model_executor", None)
                    sockets = getattr(executor, "cmd_sockets", [])
                    
                    for i, sock in enumerate(sockets):
                        worker_idx = i + 1
                        start_idx = worker_idx * local_batch_size
                        end_idx = start_idx + local_batch_size
                        
                        worker_completions = completions[start_idx:end_idx]
                        worker_log_probs = token_log_probs_list[start_idx:end_idx]
                        
                        payload = pickle.dumps((worker_completions, worker_log_probs))
                        sock.send(payload)
                    
                    completions_slice = completions[:local_batch_size]
                    token_log_probs_slice = token_log_probs_list[:local_batch_size]
                else:
                    import pickle
                    payload = self.zmq_sock.recv()
                    completions_slice, token_log_probs_slice = pickle.loads(payload)
                    
                completions_tensor = torch.tensor(completions_slice, device=prompt_ids_repeated.device, dtype=torch.long)
                token_log_probs = torch.tensor(token_log_probs_slice, device=prompt_ids_repeated.device, dtype=torch.float32)
            else:
                # Broadcast the results from the driver rank to all other ranks within the TP subgroup via TPU ICI.
                # this seems causing hang with multihost, need to investigate
                import torch.distributed as dist
                import torch_tpu
                # Ensure all TPU ops from vLLM generation are finished before running PyTorch collectives
                torch_tpu._internal.sync.synchronize(wait=True)
                dist.barrier()
                
                max_new_tokens = sampling_params.max_tokens
                batch_size_per_group = len(prompt_ids_repeated) * self.tp_size
                
                if self.is_driver:
                    comp_t = torch.tensor(completions, dtype=torch.long, device="tpu")
                    probs_t = torch.tensor(token_log_probs_list, dtype=torch.float32, device="tpu")
                else:
                    comp_t = torch.zeros((batch_size_per_group, max_new_tokens), dtype=torch.long, device="tpu")
                    probs_t = torch.zeros((batch_size_per_group, max_new_tokens), dtype=torch.float32, device="tpu")
                    
                driver_rank = self.rank - (self.rank % self.tp_size)
                dist.broadcast(comp_t, src=driver_rank, group=self.tp_group)
                dist.broadcast(probs_t, src=driver_rank, group=self.tp_group)
                
                # Slice the completions and logprobs back down to the local worker's batch size
                local_idx_in_group = self.rank % self.tp_size
                start_idx = local_idx_in_group * local_batch_size
                end_idx = start_idx + local_batch_size
                
                completions_tensor = comp_t[start_idx:end_idx]
                token_log_probs = probs_t[start_idx:end_idx]
        else:
            completions_tensor = torch.tensor(completions, device=prompt_ids_repeated.device, dtype=torch.long)
            token_log_probs = torch.tensor(token_log_probs_list, device=prompt_ids_repeated.device, dtype=torch.float32)

        completed_ids = torch.cat([prompt_ids_repeated, completions_tensor], dim=1)
        return completed_ids, token_log_probs
