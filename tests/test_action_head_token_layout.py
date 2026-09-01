import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_action_heads_module():
    constants = types.ModuleType("prismatic.vla.constants")
    constants.ACTION_DIM = 7
    constants.ACTION_TOKEN_BEGIN_IDX = 0
    constants.IGNORE_INDEX = -100
    constants.NUM_ACTIONS_CHUNK = 8
    constants.PROPRIO_DIM = 8
    constants.STOP_INDEX = 0
    constants.NUM_TOKENS = 64

    module_path = Path(__file__).parents[1] / "prismatic" / "models" / "action_heads.py"
    spec = importlib.util.spec_from_file_location("action_heads_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    previous_constants = sys.modules.get(constants.__name__)
    try:
        sys.modules[constants.__name__] = constants
        spec.loader.exec_module(module)
    finally:
        if previous_constants is None:
            sys.modules.pop(constants.__name__, None)
        else:
            sys.modules[constants.__name__] = previous_constants
    return module


action_heads = _load_action_heads_module()


@pytest.mark.parametrize("num_images", [1, 2, 3])
def test_split_preserves_all_camera_and_action_tokens(num_images):
    num_task_tokens = 256 * num_images
    task = torch.arange(num_task_tokens).reshape(1, 1, num_task_tokens, 1)
    actions = torch.arange(64).reshape(1, 1, 64, 1) + 10_000
    hidden_states = torch.cat((task, actions), dim=2)

    actual_task, actual_actions = action_heads.split_task_action_hidden_states(hidden_states)

    torch.testing.assert_close(actual_task, task)
    torch.testing.assert_close(actual_actions, actions)


@pytest.mark.parametrize("num_images", [1, 3])
def test_action_head_conditions_on_every_camera_and_exactly_64_action_tokens(num_images):
    class CaptureConditioning(torch.nn.Module):
        def forward(self, x, h_a, p, h_t):
            self.task = h_t
            self.actions = h_a
            return torch.zeros(x.shape[0], x.shape[1], 7)

    num_task_tokens = 256 * num_images
    task = torch.randn(1, 2, num_task_tokens, 8)
    actions = torch.randn(1, 2, 64, 8)
    head = action_heads.L1RegressionActionHead(input_dim=8, hidden_dim=8)
    head.model = CaptureConditioning()

    head.predict_action(
        torch.cat((task, actions), dim=2),
        proprio=torch.zeros(1, 8),
        proprio_projector=torch.nn.Identity(),
    )

    torch.testing.assert_close(head.model.task, task)
    torch.testing.assert_close(head.model.actions, actions)


@pytest.mark.parametrize(
    ("hidden_states", "num_action_tokens"),
    [(torch.empty(1, 2, 64, 4), 64), (torch.empty(1, 2, 63, 4), 64)],
)
def test_split_rejects_sequences_without_task_tokens(hidden_states, num_action_tokens):
    with pytest.raises(ValueError, match="must exceed"):
        action_heads.split_task_action_hidden_states(hidden_states, num_action_tokens)


def test_dynamic_split_keeps_legacy_action_head_state_dict_compatible():
    legacy_head = action_heads.L1RegressionActionHead(input_dim=8, hidden_dim=8, num_task_tokens=512)
    dynamic_head = action_heads.L1RegressionActionHead(input_dim=8, hidden_dim=8)

    incompatible_keys = dynamic_head.load_state_dict(legacy_head.state_dict(), strict=True)

    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []
