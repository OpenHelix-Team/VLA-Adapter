"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union
import time
import sys

import draccus
import msgpack
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image
import msgpack_numpy


# Append project root to sys.path
sys.path.append("../..")

from experiments.robot import openvla_utils
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from prismatic.util.device import resolve_device
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)


# Set up logging to display timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port
    device: str = ""                                                    # Device to run model on ("" = detect: xpu, cuda, cpu)

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                    # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                   # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    phase: str = "Inference"
    use_pro_version: bool = True



def initialize_model(cfg: DeployConfig):
    """Initialize model and associated components."""
    # The loaders in openvla_utils read this module-level device at call time
    openvla_utils.DEVICE = resolve_device(cfg.device)

    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    
    # Get number of vision patches
    NUM_PATCHES = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    cfg.num_task_tokens=NUM_PATCHES

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=14,  # 14-dimensional proprio for aloha
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Get OpenVLA processor
    processor = get_processor(cfg)


    return model, processor, action_head, proprio_projector


class MsgPackResponse(Response):
    """Custom FastAPI Response class to automatically encode response data into MessagePack."""

    media_type = "application/msgpack"

    def render(self, content: Any) -> bytes:
        return msgpack.packb(content, default=msgpack_numpy.encode, use_bin_type=True)


# === Server Interface ===
class VLAServer:
    def __init__(self, cfg: DeployConfig):
        """
        A simple server for VLA models, exposing `/act` endpoint.
        This server receives observations and instructions via MessagePack,
        and returns predicted actions in MessagePack format.
        """
        self.cfg = cfg
        (
            self.model,
            self.processor,
            self.action_head,
            self.proprio_projector,
        ) = initialize_model(cfg)
        self.resize_size = get_image_resize_size(cfg)
        set_seed_everywhere(self.cfg.seed)
        self.app = FastAPI()

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """
            Middleware to log request details including processing time.
            """
            start_time = time.time()
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000  # in milliseconds
            logging.info(f'"{request.method} {request.url.path}" {response.status_code} - {process_time:.2f}ms')
            return response

        self.app.post("/act", response_class=MsgPackResponse)(self.get_server_action)

    async def get_server_action(self, request: Request) -> Dict[str, Any]:
        """Handles a single action prediction request using MessagePack."""
        if request.headers.get("content-type") != "application/msgpack":
            raise HTTPException(
                status_code=415, detail="Unsupported Media Type. 'application/msgpack' is required."
            )
        try:
            body = await request.body()
            batch = msgpack.unpackb(body, object_hook=msgpack_numpy.decode, raw=False)
            
            # Extract unnorm_key and instruction from the batch
            unnorm_key = batch.pop("unnorm_key")
            instruction = batch.pop("instruction")

            # Update cfg with the unnorm_key from the client
            self.cfg.unnorm_key = unnorm_key

            # Use get_action to get model's prediction
            actions = get_action(
                self.cfg,
                self.model,
                batch,
                instruction,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
            )
            
            return {"actions": np.array(actions).tolist()}

        except msgpack.UnpackException:
            raise HTTPException(status_code=400, detail="Invalid MessagePack data provided.")
        except Exception:
            logging.error(traceback.format_exc())
            # Re-raise as a generic 500 error to avoid leaking implementation details.
            raise HTTPException(status_code=500, detail="An internal server error occurred.")

    def run(self) -> None:
        """Starts the Uvicorn server."""
        uvicorn.run(self.app, host=self.cfg.host, port=self.cfg.port, access_log=False, timeout_keep_alive=120)


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = VLAServer(cfg)
    server.run()


if __name__ == "__main__":
    deploy()
