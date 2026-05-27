# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

import unittest
from unittest.mock import MagicMock, patch
import torch

from torchtitan.experiments.tpu.rl.ray_orchestrator import GRPOOrchestrator

class TestGRPOOrchestrator(unittest.TestCase):
    def setUp(self):
        self.sys_argv = ["ray_train.py", "--module", "torchtitan.experiments.tpu.rl"]
        
    @patch("torchtitan.experiments.tpu.rl.ray_orchestrator.FusedWorker")
    @patch("torchtitan.experiments.tpu.rl.ray_orchestrator.ConfigManager")
    def test_setup_workers(self, mock_config_manager, mock_fused_worker):
        # Mock config
        mock_config = MagicMock()
        mock_config_manager.return_value.parse_args.return_value = mock_config
        
        # Setup orchestrator
        orchestrator = GRPOOrchestrator(
            sys_argv=self.sys_argv,
            world_size=2,
            tpu_resources={"TPU": 1},
            master_addr="127.0.0.1",
            master_port="12345",
            sb_addresses="localhost:1001,localhost:1002"
        )
        
        orchestrator.setup_workers()
        
        # Check workers created
        self.assertEqual(len(orchestrator.workers), 2)
        mock_fused_worker.options.assert_called()

if __name__ == "__main__":
    unittest.main()

