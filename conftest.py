import os
import sys
import pytest

# Prevent PyTorch from auto-loading torch_tpu in the main pytest process.
# This prevents the parent process from acquiring the TPU lock, which causes
# child processes spawned by torchrun to hang.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# 1. Duplicate Registration during Collection
try:
    from torchtitan.protocols import train_spec as train_spec_mod
    
    _orig_register_train_spec = train_spec_mod.register_train_spec
    
    def _idempotent_register_train_spec(name, train_spec=None, **kwargs):
        if name in getattr(train_spec_mod, '_extra_train_specs', {}):
            return
        
        # Call the original method matching its original signature
        if train_spec is not None:
            _orig_register_train_spec(name, train_spec)
        else:
            _orig_register_train_spec(name, **kwargs)
        
    train_spec_mod.register_train_spec = _idempotent_register_train_spec
except ImportError:
    pass

def pytest_configure(config):
    # Just inject the default device flag if not present
    has_device_flag = any(arg.startswith("--device") for arg in sys.argv)
    if not has_device_flag:
        sys.argv.append("--device=tpu")

def pytest_collection_modifyitems(config, items):
    # Parse absl flags after collection so test_utils is already imported
    try:
        from absl import flags
        absl_args = [sys.argv[0]]
        for arg in sys.argv[1:]:
            if arg.startswith("--device") or arg.startswith("--num_devices") or arg.startswith("--test_random_seed"):
                absl_args.append(arg)
        flags.FLAGS(absl_args, known_only=True)
    except ImportError:
        pass

    # 6. Forking vs. Spawning Deadlocks
    for item in items:
        is_distributed = False
        
        # Check if it's a distributed test based on class inheritance
        if getattr(item, "cls", None) is not None:
            cls_names = [c.__name__ for c in item.cls.__mro__]
            if "BaseDistributedDeviceTest" in cls_names:
                is_distributed = True
                
        # Fallback based on test name or path
        if not is_distributed and ("distributed" in item.nodeid.lower() or "dtensor" in item.nodeid.lower()):
            is_distributed = True
            
        # Dynamically apply @pytest.mark.forked marker to non-distributed (Unit/Kernel) tests
        if not is_distributed:
            item.add_marker(pytest.mark.forked)
        
        # Debugging output
        if "test_model_distributed" in item.nodeid:
            print(f"DEBUG: {item.nodeid} -> is_distributed={is_distributed}")
