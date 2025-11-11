"""
run_aloha_eval_local.py

Evaluates a model in a real-world ALOHA environment with local model deployment.
"""

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import tqdm
import torch
import json
import numpy as np

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.aloha.aloha_utils import (
    get_aloha_env,
    get_aloha_image,
    get_aloha_wrist_images,
    get_next_task_label,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = ""  # Pretrained checkpoint path

    use_l1_regression: bool = True  # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False  # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50  # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False  # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3  # Number of images in the VLA input (default: 3)
    use_proprio: bool = True  # Whether to include proprio state in input

    center_crop: bool = True  # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25  # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""  # Action un-normalization key

    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # ALOHA environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50  # Number of test rollouts
    max_steps: int = 1500  # Max number of steps per rollout
    use_relative_actions: bool = False  # Whether to use relative actions (delta joint angles)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"  # Local directory for eval logs

    seed: int = 7  # Random Seed (for reproducibility)

    save_version: str = "vla-adapter"  # version of
    use_pro_version: bool = True  # encourage to use the pro models we released.
    phase: str = "Inference"

    # fmt: on


class LocalVLAModel:
    """Local VLA model for direct inference without server."""

    def __init__(self, cfg: GenerateConfig):
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, 7)  # PROPRIO_DIM = 14

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)

        logger.info("Local VLA model loaded successfully")

    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Get action from local model."""
        instruction = observation["instruction"]

        action = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )

        return action


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint, "Must provide pretrained_checkpoint for local model deployment!"
    assert os.path.exists(cfg.pretrained_checkpoint), f"Checkpoint path {cfg.pretrained_checkpoint} does not exist!"


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    # Create run ID
    run_id = f"EVAL-LOCAL-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_aloha_image(obs)
    right_wrist_img = get_aloha_wrist_images(obs)

    # Resize images to size expected by model
    from experiments.robot.openvla_utils import resize_image_for_policy
    img_resized = resize_image_for_policy(img, resize_size)
    # left_wrist_img_resized = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        # "left_wrist_image": left_wrist_img_resized,
        "right_wrist_image": right_wrist_img_resized,
        "state": obs.observation["qpos"],
    }

    # return observation, img_resized, left_wrist_img_resized, right_wrist_img_resized
    return observation, img_resized, right_wrist_img_resized


def run_episode(
        cfg: GenerateConfig,
        env,
        task_description: str,
        local_model: LocalVLAModel,
        resize_size,
        log_file=None,
):
    """Run a single episode in the ALOHA environment."""
    # Define control frequency
    STEP_DURATION_IN_SEC = 1.0 / 50.0

    # Reset environment
    obs = env.reset()

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    curr_state = None
    replay_images = []
    replay_images_resized = []
    replay_images_left_wrist_resized = []
    replay_images_right_wrist_resized = []

    log_message("Prepare the scene, and then press Enter to begin...", log_file)
    input()

    # Reset environment again to fetch first timestep observation
    obs = env.reset()

    # Fetch initial robot state (but sleep first so that robot stops moving)
    time.sleep(2)
    curr_state = env.get_qpos()

    episode_start_time = time.time()
    total_model_query_time = 0.0

    try:
        while t < cfg.max_steps:
            # Get step start time (used to compute how much to sleep between steps)
            step_start_time = time.time()

            # Get observation
            obs = env.get_observation(t=t)

            # Save raw high camera image for replay video
            replay_images.append(obs.observation["images"]["cam_high"])

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Prepare observation
                observation, img_resized, right_wrist_resized = prepare_observation(obs, resize_size)
                observation["instruction"] = task_description

                # Save processed images for replay
                replay_images_resized.append(img_resized)
                # replay_images_left_wrist_resized.append(left_wrist_resized)
                replay_images_right_wrist_resized.append(right_wrist_resized)

                # Query model to get action
                log_message("Querying local model...", log_file)
                model_query_start_time = time.time()
                actions = local_model.get_action(observation)
                actions = actions[: cfg.num_open_loop_steps]
                total_model_query_time += time.time() - model_query_start_time
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()
            log_message("-----------------------------------------------------", log_file)
            log_message(f"t: {t}", log_file)
            log_message(f"action: {action}", log_file)

            # Execute action in environment
            if cfg.use_relative_actions:
                # Get absolute joint angles from relative action
                rel_action = action
                target_state = curr_state + rel_action
                obs = env.step(target_state.tolist())
                # Update current state (assume it is the commanded target state)
                curr_state = target_state
            else:
                obs = env.step(action.tolist())
            t += 1

            # Sleep until next timestep
            step_elapsed_time = time.time() - step_start_time
            if step_elapsed_time < STEP_DURATION_IN_SEC:
                time_to_sleep = STEP_DURATION_IN_SEC - step_elapsed_time
                log_message(f"Sleeping {time_to_sleep} sec...", log_file)
                time.sleep(time_to_sleep)

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)
        else:
            log_message(f"\nCaught exception: {e}", log_file)

    episode_end_time = time.time()

    # Get success feedback from user
    user_input = input("Success? Enter 'y' or 'n': ")
    success = True if user_input.lower() == 'y' else False

    # Calculate episode statistics
    episode_stats = {
        "success": success,
        "total_steps": t,
        "model_query_time": total_model_query_time,
        "episode_duration": episode_end_time - episode_start_time,
    }

    return (
        episode_stats,
        replay_images,
        replay_images_resized,
        replay_images_left_wrist_resized,
        replay_images_right_wrist_resized,
    )


def save_episode_videos(
        replay_images,
        replay_images_resized,
        replay_images_left_wrist,
        replay_images_right_wrist,
        episode_idx,
        success,
        task_description,
        log_file=None,
):
    """Save videos of the episode from different camera angles."""
    # Save main replay video
    save_rollout_video(replay_images, episode_idx, success=success, task_description=task_description,
                       log_file=log_file)

    # Save processed view videos
    save_rollout_video(
        replay_images_resized,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="resized",
    )
    save_rollout_video(
        replay_images_left_wrist,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="left_wrist_resized",
    )
    save_rollout_video(
        replay_images_right_wrist,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="right_wrist_resized",
    )


@draccus.wrap()
def eval_aloha_local(cfg: GenerateConfig) -> None:
    """Main function to evaluate a trained policy in a real-world ALOHA environment with local model."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Load local model
    log_message("Loading local VLA model...", log_file)
    local_model = LocalVLAModel(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Get ALOHA environment
    env = get_aloha_env()

    # Initialize task description
    task_description = ""

    # Start evaluation
    num_rollouts_completed, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_rollouts_planned)):
        # Get task description from user
        task_description = get_next_task_label(task_description)
        log_message(f"\nTask: {task_description}", log_file)

        log_message(f"Starting episode {num_rollouts_completed + 1}...", log_file)

        # Run episode
        episode_stats, replay_images, replay_images_resized, replay_images_left_wrist, replay_images_right_wrist = (
            run_episode(cfg, env, task_description, local_model, resize_size, log_file)
        )

        # Update counters
        num_rollouts_completed += 1
        if episode_stats["success"]:
            total_successes += 1

        # Save videos
        save_episode_videos(
            replay_images,
            replay_images_resized,
            replay_images_left_wrist,
            replay_images_right_wrist,
            num_rollouts_completed,
            episode_stats["success"],
            task_description,
            log_file,
        )

        # Log results
        log_message(f"Success: {episode_stats['success']}", log_file)
        log_message(f"# episodes completed so far: {num_rollouts_completed}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / num_rollouts_completed * 100:.1f}%)", log_file)
        log_message(f"Total model query time: {episode_stats['model_query_time']:.2f} sec", log_file)
        log_message(f"Total episode elapsed time: {episode_stats['episode_duration']:.2f} sec", log_file)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_rollouts_completed) if num_rollouts_completed > 0 else 0

    # Log final results
    log_message("\nFinal results:", log_file)
    log_message(f"Total episodes: {num_rollouts_completed}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_aloha_local()