import importlib.metadata
import sys

# Monkey-patch importlib.metadata.entry_points to hide torch_tpu
# This prevents PyTorch from auto-loading torch_tpu in the main pytest process,
# which would acquire the TPU lock and cause the spawned child processes to hang.
_original_entry_points = importlib.metadata.entry_points

def _mock_entry_points(*args, **kwargs):
    eps = _original_entry_points(*args, **kwargs)
    if 'group' in kwargs and kwargs['group'] == 'torch.backends':
        # Filter out torch_tpu
        return [ep for ep in eps if ep.name != 'torch_tpu']
    if len(args) > 0 and args[0] == 'torch.backends':
        return [ep for ep in eps if ep.name != 'torch_tpu']
    
    # In newer python versions, entry_points() returns a SelectableGroups object
    if hasattr(eps, 'select'):
        class MockSelectableGroups:
            def __init__(self, original):
                self._original = original
            def select(self, **kwargs):
                selected = self._original.select(**kwargs)
                if kwargs.get('group') == 'torch.backends':
                    return [ep for ep in selected if ep.name != 'torch_tpu']
                return selected
            def get(self, key, default=None):
                res = self._original.get(key, default)
                if key == 'torch.backends' and res is not None:
                    return [ep for ep in res if ep.name != 'torch_tpu']
                return res
            def __getitem__(self, key):
                res = self._original[key]
                if key == 'torch.backends':
                    return [ep for ep in res if ep.name != 'torch_tpu']
                return res
        return MockSelectableGroups(eps)
    
    return eps

importlib.metadata.entry_points = _mock_entry_points

def pytest_ignore_collect(collection_path, config):
    if collection_path.name == 'test_utils.py':
        return True
