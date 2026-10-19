"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)
from prismatic.util.device import default_device, manual_seed_all

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = default_device()

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.8f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> torch.nn.Module:
    """
    Load and initialize model for evaluation based on configuration.

    Args:
        cfg: Configuration object with model parameters
        wrap_diffusion_policy_for_droid: Whether to wrap diffusion policy for DROID

    Returns:
        torch.nn.Module: The loaded model

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    use_minivlm: bool = False,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Query the model to get action predictions.

    Args:
        cfg: Configuration object with model parameters
        model: The loaded model
        obs: Observation dictionary
        task_label: Text description of the task
        processor: Model processor for inputs
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        Union[List[np.ndarray], np.ndarray]: Predicted actions

    Raises:
        ValueError: If model family is not supported
    """
    with torch.no_grad():
        if cfg.model_family == "openvla":
            action = get_vla_action(
                cfg=cfg,
                vla=model,
                processor=processor,
                obs=obs,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
                use_minivlm=use_minivlm
            )
        else:
            raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


# -------- Msgpack for VLA: Start ----------

import msgpack
import msgpack_numpy
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class MsgPackHttpClientPolicy:
    """A client policy for communicating with a VLA server using MessagePack."""

    def __init__(self, host: str, port: int=None):
        """
        Initializes the client.

        Args:
            host (str): The server host address.
            port (int): The server port.
        """
        protocol = "https" if "nat-notebook-inspire" in host or "ngrok" in host else "http"
        if host.startswith("http"):
            base_url = host
        else:
            base_url = f"{protocol}://{host}:{port}"
        
        self.infer_url = f"{base_url.rstrip('/')}/act"
        self.session = requests.Session()
        # Robust retries for occasional connection resets from server
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.2,
            status_forcelist=(502, 503, 504),
            raise_on_status=False,
            allowed_methods=frozenset(["POST", "GET"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"Content-Type": "application/msgpack"})
        print(f"Standalone MsgPack HTTP Client configured for: {self.infer_url}")

    def infer(self, observation: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Sends an observation to the server and returns the predicted action.

        Args:
            observation (Dict[str, Any]): The observation dictionary.

        Returns:
            Dict[str, Any]: The action dictionary from the server.
        """
        packed_observation = msgpack.packb(observation, default=msgpack_numpy.encode, use_bin_type=True)
        try:
            response = self.session.post(self.infer_url, data=packed_observation, timeout=30)
            response.raise_for_status()
            return msgpack.unpackb(response.content, object_hook=msgpack_numpy.decode, raw=False)
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference request failed: {e}")
            # Propagate exception to let the main loop handle it
            raise e
        except msgpack.UnpackException as e:
            logger.error(f"Failed to unpack server response: {e}")
            raise e
        
        
# -------- Msgpack for VLA: End ----------