"""Device selection for the inference path (XPU > CUDA > CPU).

Everything except the last test runs without an accelerator, by faking which backends exist.
The parity test needs an XPU and uses a tiny random-weight action head, so no checkpoint is needed.
"""

import importlib.util
import unittest
from unittest import mock

import torch

from prismatic.util import device as dev


def _fake_availability(*, xpu: bool, cuda: bool):
    def fake(device_type: str) -> bool:
        return {"xpu": xpu, "cuda": cuda}.get(device_type, device_type == "cpu")

    return mock.patch.object(dev, "is_available", side_effect=fake)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _fake_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.default_device(), torch.device("xpu", 0))

    def test_cuda_default_is_unchanged(self):
        with _fake_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device(), torch.device("cuda:0"))

    def test_cpu_without_accelerator(self):
        with _fake_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device(), torch.device("cpu"))
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _fake_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device(torch.device("xpu:1")), torch.device("xpu:1"))

    def test_empty_string_means_detect(self):
        # DeployConfig.device defaults to "" so the server detects the backend
        with _fake_availability(xpu=True, cuda=False):
            self.assertEqual(dev.resolve_device(""), torch.device("xpu", 0))


class SeedingTest(unittest.TestCase):
    def test_seed_goes_to_detected_backend_only(self):
        with _fake_availability(xpu=True, cuda=False):
            with mock.patch.object(torch.xpu, "manual_seed_all") as xpu_seed, mock.patch.object(
                torch.cuda, "manual_seed_all"
            ) as cuda_seed:
                dev.manual_seed_all(3)
                xpu_seed.assert_called_once_with(3)
                cuda_seed.assert_not_called()
        with _fake_availability(xpu=False, cuda=True):
            with mock.patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
                dev.manual_seed_all(3)
                cuda_seed.assert_called_once_with(3)

    def test_seed_is_a_noop_on_cpu(self):
        with _fake_availability(xpu=False, cuda=False):
            with mock.patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
                dev.manual_seed_all(3)
                cuda_seed.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "tensorflow is needed to import the eval utils")
class EvalUtilsTest(unittest.TestCase):
    def test_module_devices_follow_detection(self):
        from experiments.robot import openvla_utils, robot_utils

        self.assertEqual(openvla_utils.DEVICE, dev.default_device())
        self.assertEqual(robot_utils.DEVICE, dev.default_device())

    def test_set_seed_everywhere_seeds_through_device_module(self):
        from experiments.robot import robot_utils

        with mock.patch.object(robot_utils, "manual_seed_all") as seed_all:
            robot_utils.set_seed_everywhere(5)
            seed_all.assert_called_once_with(5)


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_action_head_cpu_matches_xpu(self):
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

        dim, task_tokens, num_layers = 32, 4, 24
        torch.manual_seed(0)
        head = L1RegressionActionHead(
            input_dim=dim, hidden_dim=dim, action_dim=ACTION_DIM, num_task_tokens=task_tokens, use_pro_version=True
        ).eval()
        proprio_projector = ProprioProjector(dim, PROPRIO_DIM).eval()
        for module in (head, proprio_projector):
            for param in module.parameters():  # no zero-initialised layer may hide a mismatch
                torch.nn.init.normal_(param, std=0.1)
        hidden = torch.randn(1, num_layers + 1, task_tokens + ACTION_DIM * NUM_ACTIONS_CHUNK, dim)
        proprio = torch.randn(1, PROPRIO_DIM)

        def run(device):
            h, p = head.to(device, torch.bfloat16), proprio_projector.to(device, torch.bfloat16)
            with torch.no_grad():
                return h.predict_action(hidden.to(device, torch.bfloat16), proprio.to(device), p).float().cpu()

        cpu_out, xpu_out = run("cpu"), run("xpu")
        self.assertTrue(torch.isfinite(xpu_out).all())
        torch.testing.assert_close(xpu_out, cpu_out, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
