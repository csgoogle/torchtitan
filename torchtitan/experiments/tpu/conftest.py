import sys
from absl import flags
import pytest

# Ignore utility and base class files during collection
collect_ignore = [
    "test_utils.py",
    "base_device_test.py",
    "base_distributed_device_test.py",
]

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="tpu", help="Accelerator device to test")
    parser.addoption("--num_devices", action="store", default=None, type=int, help="Number of devices")

def pytest_configure(config):
    import os
    # Add project root to sys.path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Parse absl flags
    device = config.getoption("--device")
    num_devices = config.getoption("--num_devices")
    
    # Import modules that define absl flags
    import torchtitan.experiments.tpu.test_utils
    
    # We parse fake argv to absl flags to avoid pytest args confusing absl
    # This also ensures flags are initialized before any test runs.
    argv = ["conftest.py", f"--device={device}"]
    if num_devices is not None:
        argv.append(f"--num_devices={num_devices}")
        
    try:
        flags.FLAGS(argv)
    except flags.DuplicateFlagError:
        # Flags might already be initialized in some environments
        pass

@pytest.fixture(autouse=True)
def setup_absl_flags(request):
    """Ensure absl flags are consistent with pytest options for each test."""
    device = request.config.getoption("--device")
    num_devices = request.config.getoption("--num_devices")
    
    # Update flags if they differ (e.g. if modified by previous tests)
    # Note: we use the Enum class for device if it's an enum flag
    from torchtitan.experiments.tpu import accelerator_device_type as device_type
    
    # Convert string to enum if necessary
    if isinstance(device, str):
        try:
            device = device_type.AcceleratorDeviceType(device.lower())
        except ValueError:
            pass

    # Use flags.FLAGS to set values correctly
    flags.FLAGS.device = device
    if num_devices is not None:
        flags.FLAGS.num_devices = num_devices

