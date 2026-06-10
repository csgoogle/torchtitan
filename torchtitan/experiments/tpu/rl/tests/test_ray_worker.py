# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

import unittest
from unittest.mock import MagicMock, patch
import torch

from torchtitan.experiments.tpu.rl.ray_worker import FusedWorker
OriginalFusedWorker = FusedWorker.__ray_metadata__.modified_class

class TestFusedWorker(unittest.TestCase):
    def setUp(self):
        self.sys_argv = ["ray_train.py", "--module", "torchtitan.experiments.tpu.rl", "--config", "grpo_qwen3_0_6b"]
        
    @patch("torchtitan.experiments.tpu.rl.ray_worker.ConfigManager")
    def test_init_worker(self, mock_config_manager):
        mock_config = MagicMock()
        mock_config_manager.return_value.parse_args.return_value = mock_config
        
        worker = OriginalFusedWorker(sys_argv=self.sys_argv)
        self.assertEqual(worker.sys_argv, self.sys_argv)
        
        # Test heartbeat
        self.assertTrue(worker.heartbeat())

    @patch("torchtitan.experiments.tpu.rl.ray_worker.ConfigManager")
    def test_load_next_batch(self, mock_config_manager):
        worker = OriginalFusedWorker(sys_argv=self.sys_argv)
        worker.device = torch.device("cpu")
        worker.torch = torch
        
        prompt_ids_list = [[1, 2, 3], [4, 5, 6]]
        worker.load_next_batch(prompt_ids_list)
        
        expected_tensor = torch.tensor(prompt_ids_list, dtype=torch.long, device=torch.device("cpu"))
        self.assertTrue(torch.allclose(worker.current_prompt_ids, expected_tensor))

if __name__ == "__main__":
    unittest.main()
