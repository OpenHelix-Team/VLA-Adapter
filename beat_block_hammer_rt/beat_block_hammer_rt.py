from typing import Iterator, Tuple, Any
import json
import random
import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import cv2

# 禁用 GCS 以防止网络相关报错
tfds.core.utils.gcs_utils._is_gcs_disabled = True

def get_random_seen_instruction(file_path):
    """辅助函数：从 JSON 读取指令"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'seen' in data and isinstance(data['seen'], list):
            seen_instructions = data['seen']
            if not seen_instructions:
                # print("错误：'seen' 列表为空。")
                return "do something"
            return random.choice(seen_instructions)
        else:
            # print(f"错误：在文件 '{file_path}' 中没有找到 'seen' 键。")
            return "do something"

    except Exception as e:
        print(f"Error reading instruction {file_path}: {e}")
        return "do something"


class BeatBlockHammerRt(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for beat_block_hammer dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(240, 320, 3), # 注意：这里定义的是 256
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'left_wrist_image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left wrist camera RGB observation.',
                        ),
                        'right_wrist_image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'low_cam_image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Lower camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot arm action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Yields episodes for list of data paths."""
        
        for episode_path in paths:
            if not os.path.exists(episode_path):
                continue

            try:
                # Load raw data
                with h5py.File(episode_path, "r") as F:
                    states = F["/joint_action/vector"][()]
                    actions = states
                    
                    images_bytes = F["/observation/head_camera/rgb"][()]
                    image_left_bytes = F["/observation/left_camera/rgb"][()]
                    image_right_bytes = F["/observation/right_camera/rgb"][()]
                    image_low_bytes = F["/observation/front_camera/rgb"][()] # 注意这里加了 [()]

                    images, left_wrist_images, right_wrist_images, low_cam_images = [], [], [], []
                    
                    # 遍历所有帧
                    for img, img_left, img_right, img_low in zip(images_bytes, image_left_bytes, image_right_bytes, image_low_bytes):
                        # 处理主视图
                        image_np_array = np.frombuffer(img, dtype=np.uint8)
                        image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
                        # 【重要】Resize 到 256x256 以匹配 _info 中的定义
                        # image = cv2.resize(image, (256, 256)) 
                        images.append(image)
                        
                        # 处理左手腕
                        image_left_np_array = np.frombuffer(img_left, dtype=np.uint8)
                        image_left = cv2.imdecode(image_left_np_array, cv2.IMREAD_COLOR)
                        # image_left = cv2.resize(image_left, (256, 256))
                        left_wrist_images.append(image_left)
                        
                        # 处理右手腕
                        image_right_np_array = np.frombuffer(img_right, dtype=np.uint8)
                        image_right = cv2.imdecode(image_right_np_array, cv2.IMREAD_COLOR)
                        # image_right = cv2.resize(image_right, (256, 256))
                        right_wrist_images.append(image_right)
                        
                        # 处理低位视图
                        image_low_np_array = np.frombuffer(img_low, dtype=np.uint8)
                        image_low = cv2.imdecode(image_low_np_array, cv2.IMREAD_COLOR)
                        # image_low = cv2.resize(image_low, (256, 256))
                        low_cam_images.append(image_low)
                    
                    images = np.array(images, dtype=np.uint8)
                    left_wrist_images = np.array(left_wrist_images, dtype=np.uint8)
                    right_wrist_images = np.array(right_wrist_images, dtype=np.uint8)
                    low_cam_images = np.array(low_cam_images, dtype=np.uint8)

                # Get language instruction
                episode_filename = os.path.basename(episode_path)
                # Construct the corresponding instruction file path
                # 假设指令文件名格式对应：episode_0.hdf5 -> episode_0.json 或 episode0.hdf5 -> episode0.json
                instruction_filename = episode_filename.replace('.hdf5', '.json')
                instruction_path = os.path.join('/home/ruihengwang/vla/RoboTwin/data/beat_block_hammer/demo_clean/instructions', instruction_filename)
                
                command = get_random_seen_instruction(instruction_path)

                # Assemble episode
                episode = []
                for i in range(actions.shape[0]):
                    episode.append({
                        'observation': {
                            'image': images[i],
                            'left_wrist_image': left_wrist_images[i],
                            'right_wrist_image': right_wrist_images[i],
                            'low_cam_image': low_cam_images[i],
                            'state': np.asarray(states[i], np.float32),
                        },
                        'action': np.asarray(actions[i], dtype=np.float32),
                        'discount': 1.0,
                        'reward': float(i == (actions.shape[0] - 1)),
                        'is_first': i == 0,
                        'is_last': i == (actions.shape[0] - 1),
                        'is_terminal': i == (actions.shape[0] - 1),
                        'language_instruction': command,
                    })

                # Create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path
                    }
                }

                # Yield the result
                yield episode_path, sample

            except Exception as e:
                print(f"Error processing {episode_path}: {e}")
                
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # 定义基础路径
        base_path = "/home/ruihengwang/vla/RoboTwin/data/beat_block_hammer/demo_clean/data"
        
        # 使用 glob 获取所有 .hdf5 文件 (不区分 train/val，全部读取)
        # 这样就不受 range(950) 的限制，有多少读多少
        all_files = sorted(glob.glob(os.path.join(base_path, "*.hdf5")))
        
        print(f"=====Total episodes found: {len(all_files)}=========")

        return {
            'train': self._generate_examples(paths=all_files),
        }