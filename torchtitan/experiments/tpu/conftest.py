import importlib.metadata
import sys

# Monkey-patch importlib.metadata.entry_points to hide torch_tpu
# This prevents PyTorch from auto-loading torch_tpu in the main pytest process,
# which would acquire the TPU lock and cause the spawned child processes to hang.
_original_entry_points = importlib.metadata.entry_points

def _mock_entry_points(*args, **kwargs):
    eps = _original_entry_points(*args, **kwargs)
    
    if isinstance(eps, dict):
        # Python 3.9 (dict) or Python 3.10/3.11 (SelectableGroups)
        res = dict(eps)
        if 'torch.backends' in res:
            res['torch.backends'] = [ep for ep in res['torch.backends'] if getattr(ep, 'name', '') != 'torch_tpu']
        try:
            return type(eps)(res)
        except Exception:
            return res
            
    if isinstance(eps, (list, tuple)):
        # Python 3.12+ (EntryPoints tuple) or list (from older Python select)
        filtered = [ep for ep in eps if not (getattr(ep, 'group', '') == 'torch.backends' and getattr(ep, 'name', '') == 'torch_tpu')]
        try:
            return type(eps)(filtered)
        except Exception:
            return filtered
            
    return eps

importlib.metadata.entry_points = _mock_entry_points

def pytest_ignore_collect(collection_path, config):
    if collection_path.name == 'test_utils.py':
        return True

