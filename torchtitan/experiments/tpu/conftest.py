import sys
import importlib.metadata
import pytest

# Monkey-patch importlib.metadata.entry_points to hide torch_tpu
# This prevents PyTorch from auto-loading torch_tpu in the main pytest process,
# which would acquire the TPU lock and cause the spawned child processes to hang.
_original_entry_points = importlib.metadata.entry_points

def _mock_entry_points(*args, **kwargs):
    eps = _original_entry_points(*args, **kwargs)
    if isinstance(eps, dict):
        res = dict(eps)
        if 'torch.backends' in res:
            res['torch.backends'] = [ep for ep in res['torch.backends'] if getattr(ep, 'name', '') not in ('torch_tpu', 'tpu')]
        try:
            return type(eps)(res)
        except Exception:
            return res
    if hasattr(eps, 'select') or isinstance(eps, (list, tuple)):
        filtered = [ep for ep in eps if not (getattr(ep, 'group', '') == 'torch.backends' and getattr(ep, 'name', '') in ('torch_tpu', 'tpu') )]
        try:
            return type(eps)(filtered)
        except Exception:
            return filtered
    return eps

importlib.metadata.entry_points = _mock_entry_points

def pytest_configure(config):
    try:
        from absl import flags
        # Inject --device=tpu if not present
        if not any(arg.startswith("--device") for arg in sys.argv):
            sys.argv.append("--device=tpu")
        
        absl_args = [sys.argv[0]]
        for arg in sys.argv[1:]:
            if arg.startswith("--device") or arg.startswith("--num_devices") or arg.startswith("--test_random_seed"):
                absl_args.append(arg)
        flags.FLAGS(absl_args, known_only=True)
        import torchtitan.experiments.tpu.test_utils
        sys.argv = ['pytest'] + sys.argv[1:]
        flags.FLAGS(sys.argv, known_only=True)
        flags.FLAGS.mark_as_parsed()
    except ImportError:
        pass

def pytest_collection_modifyitems(config, items):
    for item in items:
        # Check if the test is distributed (spawns workers)
        is_distributed = False
        if getattr(item, "cls", None) is not None:
            cls_names = [c.__name__ for c in item.cls.__mro__]
            if "BaseDistributedDeviceTest" in cls_names:
                is_distributed = True
        
        if not is_distributed and ("distributed" in item.nodeid.lower() or "dtensor" in item.nodeid.lower()):
            is_distributed = True
            
        # To avoid TPU lock deadlocks between tests, run non-distributed tests in a forked process
        if not is_distributed:
            item.add_marker(pytest.mark.forked)

def pytest_ignore_collect(collection_path, config):
    if collection_path.name == 'test_utils.py':
        return True
