import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
from pyquaternion import Quaternion

from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import Recorder, ImageRecorder

# from base_recorder import BaseRecorder
# from scan_recorder import SCANRecorder
# from imu_recorder import IMURecorder

from robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
# import pyrealsense2 as rs
# from dynamixel_client import DynamixelClient

import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [right_arm_qpos (6),             # absolute joint position
                        right_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ right_arm_qpos (6),          # absolute joint position
                                        right_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, setup_robots=True, setup_base=False):
        # self.puppet_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
        #                                                robot_name=f'puppet_right', init_node=init_node)
        self.puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                        robot_name=f'puppet_right', init_node=init_node)

        if setup_robots:
            self.setup_robots()

        # #if setup_base:
        #     self.setup_base()


        self.recorder_right = Recorder('right', init_node=False)
        # self.base_recorder = BaseRecorder(init_node=False)
        # self.scan_recorder = SCANRecorder(init_node=False)  # 雷达scan
        # self.imu_recorder = IMURecorder(init_node=False)  # imu
        self.image_recorder = ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        setup_puppet_bot(self.puppet_bot_right)

    def get_qpos(self):
        right_qpos_raw = self.recorder_right.qpos
        right_arm_qpos = right_qpos_raw[:6]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])]  # this is position not joint
        return np.concatenate([right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        right_qvel_raw = self.recorder_right.qvel
        right_arm_qvel = right_qvel_raw[:6]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        right_effort_raw = self.recorder_right.effort
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([right_robot_effort])

    # cam
    def get_images(self):
        return self.image_recorder.get_images()  # noetic

    # -------------------------------------------
    # def get_base_vel(self):
    #     return self.base_recorder.get_vel()

    # 雷达scan
    # def get_scan_vel(self):
    #     return self.scan_recorder.get_scan_vel()

    # 雷达scan
    # def get_imu_vel(self):
    #     return self.imu_recorder.get_imu_vel()

    # def get_tracer_vel(self):
    #     linear_vel, angular_vel = self.tracer.GetLinearVelocity(), self.tracer.GetAngularVelocity()
    #     return np.array([linear_vel, angular_vel])


    def set_gripper_pose(self, right_gripper_desired_pos_normalized):
        right_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        move_arms([self.puppet_bot_right], [reset_position], move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        move_grippers([self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN], move_time=0.5)
        move_grippers([self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] , move_time=1)

    def _get_obs(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_observation(self, t=0):
        step_type = dm_env.StepType.FIRST if t == 0 else dm_env.StepType.MID
        return dm_env.TimeStep(
            step_type=step_type,
            reward=self.get_reward(),
            discount=None,
            observation=self._get_obs()
        )

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action, base_action=None, get_tracer_vel=False, get_obs=True):
        state_len = int(len(action))
        right_action = action[:state_len]

        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        self.set_gripper_pose(right_action[-1])
        #if base_action is not None:
            # linear_vel_limit = 1.5
            # angular_vel_limit = 1.5
            # base_action_linear = np.clip(base_action[0], -linear_vel_limit, linear_vel_limit)
            # base_action_angular = np.clip(base_action[1], -angular_vel_limit, angular_vel_limit)
            # base_action_linear, base_action_angular = base_action
            # self.tracer.SetMotionCommand(linear_vel=base_action_linear, angular_vel=base_action_angular)
        # time.sleep(DT)
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

def get_action(master_bot_right):
    action = np.zeros(7) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action

# def get_base_action():



def make_real_env(init_node, setup_robots=True, setup_base=False):
    env = RealEnv(init_node, setup_robots, setup_base)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_right_wrist'

    # source of data
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=True)

    setup_master_bot(master_bot_right)


    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()

