import sys
import os
import subprocess
import time
from pathlib import Path
import pytest
import absl.flags

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------------------------------------------------------------
# Pytest / ABSL Test Isolation Wrapper
# -------------------------------------------------------------------------
# Pytest's collection phase imports all modules sequentially, which clashes
# with absl.flags duplicate detection and TrainSpec duplicate registrations.
# We mock these during collection to ensure smooth discovery.

# 1. Mock ABSL flag duplicate detection
original_define_flag = absl.flags._defines.DEFINE_flag
def mock_define_flag(*args, **kwargs):
    try:
        return original_define_flag(*args, **kwargs)
    except absl.flags.DuplicateFlagError:
        pass
absl.flags._defines.DEFINE_flag = mock_define_flag

# 2. Mock TrainSpec registration
import torchtitan.protocols.train_spec as ts_module
original_register = ts_module.register_train_spec
def mock_register(name, train_spec):
    if name not in ts_module._extra_train_specs:
        original_register(name, train_spec)
ts_module.register_train_spec = mock_register

# 3. Handle required Pytest / ABSL flags
def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="tpu", help="Device to test on")
    parser.addoption("--num_devices", action="store", default=None, help="Number of devices")

absl_args = ["pytest"] + [arg for arg in sys.argv[1:] if arg.startswith("--device=") or arg.startswith("--num_devices=")]
if not any(arg.startswith("--device=") for arg in absl_args):
    absl_args.append("--device=tpu")

import torchtitan.experiments.tpu.test_utils as test_utils
absl.flags.FLAGS(absl_args)

# -------------------------------------------------------------------------
# Test Execution Interception
# -------------------------------------------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Bypasses Pytest's runner to spawn isolated subprocesses matching `python -m`"""
    if "torchtitan/experiments/tpu" not in str(item.fspath):
        return None
    
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    call_setup = pytest.CallInfo.from_call(lambda: item.ihook.pytest_runtest_setup(item=item), "setup")
    item.ihook.pytest_runtest_makereport(item=item, call=call_setup)
    
    if call_setup.excinfo is None:
        def run_subprocess():
            rel_path = Path(item.fspath).relative_to(project_root)
            module_name = str(rel_path.with_suffix("")).replace("/", ".")
            test_case = f"{item.parent.name}.{item.name}"
            
            env = os.environ.copy()
            env["PYTHONPATH"] = str(item.config.rootdir)
            
            cmd = [sys.executable, "-m", module_name, test_case, f"--device={item.config.getoption('--device')}"]
            num_devs = item.config.getoption("--num_devices")
            if num_devs is not None:
                cmd.append(f"--num_devices={num_devs}")
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            time.sleep(1.5)  # Allow TPU drivers to release /dev/vfio locks
            
            if result.returncode != 0:
                raise Exception(f"ABSL test failed:\nCommand: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        call_call = pytest.CallInfo.from_call(run_subprocess, "call")
        report = item.ihook.pytest_runtest_makereport(item=item, call=call_call)
        item.ihook.pytest_runtest_logreport(report=report)
        
    call_teardown = pytest.CallInfo.from_call(lambda: item.ihook.pytest_runtest_teardown(item=item, nextitem=nextitem), "teardown")
    item.ihook.pytest_runtest_makereport(item=item, call=call_teardown)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    
    return True