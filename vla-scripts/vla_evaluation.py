import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from prismatic.util.device import default_device
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


OPENVLA_IMAGE_SIZE = 224


def get_openvla_prompt(instruction: str, tokenized_action: str = None) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()



def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image



def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9
    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )



class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(self, model, proprio_projector, noisy_action_projector, action_head, processor, use_x0_prediction=False):
        super().__init__()

        self.device = default_device()
        self.processor = processor
        self.OFT = model
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.action_head = action_head
        self.use_x0_prediction = use_x0_prediction

        # Set x0 prediction flag in action head if using diffusion
        if self.action_head is not None and hasattr(self.action_head, 'use_x0_prediction'):
            self.action_head.use_x0_prediction = use_x0_prediction

        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        self.action = None
        self.hidden_states = None
        self.obs_buffer = None

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]

        # Dataset statics (rougnly computed with 10k samples in CALVIN)
        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0

        self.hist_action = []

        
    def reset(self,):
        """
        This is called
        """

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action = []


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            instruction: embedded language goal
        Returns:
            action: predicted action
        """
        processed_images = []
        image = obs["rgb_obs"]['rgb_static']  # {ndarray: (200, 200, 3)}
        gripper_image = obs["rgb_obs"]['rgb_gripper']  # {ndarray: (84, 84, 3)}
        # gripper_image1 = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(self.dual_sys.device)

        # tactile_image = None
        # tactile_image = torch.from_numpy(obs["rgb_obs"]['rgb_tactile']).permute(2,0,1).unsqueeze(0).to(self.dual_sys.device, dtype=torch.float) / 255
        # depth_image = torch.from_numpy(obs["depth_obs"]['depth_static']).unsqueeze(0).to(self.dual_sys.device) - self.depth_min / (self.depth_max - self.depth_min)
        # depth_gripper = torch.from_numpy(obs["depth_obs"]['depth_gripper']).unsqueeze(0).to(self.dual_sys.device) - self.gripper_depth_min / (self.gripper_depth_max - self.gripper_depth_min)

        # image = image[::-1, ::-1]
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image_resize = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image_resize).convert("RGB")
        pil_image = center_crop_image(pil_image)

        # gripper_image = gripper_image[::-1, ::-1]
        check_image_format(gripper_image)
        if gripper_image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            gripper_image = resize_image_for_policy(gripper_image, OPENVLA_IMAGE_SIZE)
        gripper_pil_image = Image.fromarray(gripper_image).convert("RGB")
        gripper_pil_image = center_crop_image(gripper_pil_image)

        processed_images.append(pil_image)
        processed_images.append(gripper_pil_image)
        primary_image = processed_images.pop(0)
        # prompt = get_openvla_prompt(instruction)
        prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {instruction.lower()}?<|im_end|>\n<|im_start|>assistant\n'
        
        inputs = self.processor(prompt, primary_image).to(self.OFT.device, dtype=torch.bfloat16)
        all_wrist_inputs = [self.processor(prompt, processed_images).to(self.OFT.device, dtype=torch.bfloat16)]
        # inputs = self.processor(prompt, Image.fromarray(image)).to(self.OFT.device, dtype=torch.bfloat16)
        # all_wrist_inputs = [self.processor(prompt, Image.fromarray(gripper_image)).to(self.OFT.device, dtype=torch.bfloat16)]

        primary_pixel_values = inputs["pixel_values"]
        all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # proprio_state = obs['robot_obs'][-8:]
        
        proprio_state = np.concatenate([obs['robot_obs'][:7], obs['robot_obs'][-1:]])  # EE position (3), EE orientation in euler angles (3), gripper width (1), joint positions (7), gripper action (1)
        proprio_norm_stats = self.OFT.norm_stats['calvin_abc_rlds']['proprio']
        # proprio_norm_stats = self.OFT.norm_stats['calvin']['proprio']

        obs["state"] = normalize_proprio(proprio_state, proprio_norm_stats)
        proprio_state = obs["state"]

        # state = torch.from_numpy(obs['robot_obs']).to(self.dual_sys.device, dtype=torch.float)
        # state = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)\
        with torch.no_grad(): 
            action, _ = self.OFT.predict_action(
                **inputs,
                unnorm_key="calvin_abc_rlds",
                # unnorm_key="calvin",
                do_sample=False,
                proprio=proprio_state,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                noisy_action_projector=self.noisy_action_projector,
                use_film=False,
            )



        action[:,-1] = 1 - action[:,-1]
        

        return [action[i] for i in range(min(len(action), 8))]
