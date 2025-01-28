import gymnasium as gym
import mujoco.viewer
import numpy as np
import mujoco
from gymnasium import spaces

class ArmEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('scene5.xml')
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        gripper_geom_names = [
            "fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4",
            "moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4"
        ]

        self.gripper_geom_ids = []
        for name in gripper_geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1:
                raise ValueError(f"Geometry '{name}' not found in model")
            self.gripper_geom_ids.append(geom_id)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")

        self.ctrl_ranges = []
        for name in ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.ctrl_ranges.append(self.model.actuator_ctrlrange[actuator_id])

        self.max_steps = 500
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Llama a la implementación de Gymnasium
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        obs = self._get_obs()
        info = {}  # Gymnasium requiere devolver (obs, info)
        return obs, info

    def step(self, action):
        scaled_action = np.zeros_like(action)
        for i in range(6):
            low, high = self.ctrl_ranges[i]
            scaled_action[i] = low + (action[i] + 1) * (high - low) / 2
        
        self.data.ctrl[:] = scaled_action
        mujoco.mj_step(self.model, self.data)

        is_grasped = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == self.object_geom_id and contact.geom2 in self.gripper_geom_ids) or \
               (contact.geom2 == self.object_geom_id and contact.geom1 in self.gripper_geom_ids):
                is_grasped = True
                break

        obs = self._get_obs(is_grasped)
        reward = self._compute_reward(obs, scaled_action, is_grasped)
        done = self._check_done(obs[12:15])
        truncated = False  # Gymnasium requiere `done` y `truncated` por separado
        info = {}

        return obs, reward, done, truncated, info

    def _get_obs(self, is_grasped=False):
        joint_pos = self.data.qpos[:6]
        joint_vel = self.data.qvel[:6]
        object_pos = self.data.body("box").xpos
        object_vel = self.data.body("box").cvel[3:]
        gripper_pos = self.data.body("Fixed_Jaw").xpos
        target_pos = np.array([0.4, 0, 0.1])

        obs = np.concatenate([
            joint_pos,
            joint_vel,
            object_pos,
            object_vel,
            gripper_pos,
            target_pos - object_pos,
            gripper_pos - object_pos,
            [float(is_grasped)]
        ])
        return obs

    def _compute_reward(self, obs, action, is_grasped):
        object_pos = obs[12:15]
        target_pos = np.array([0.4, 0, 0.1])
        distance = np.linalg.norm(object_pos - target_pos)

        in_x = 0.28 <= object_pos[0] <= 0.52
        in_y = -0.12 <= object_pos[1] <= 0.12
        in_z = object_pos[2] >= -0.02
        is_success = in_x and in_y and in_z

        reward = (
            -0.1 * distance
            + 0.5 * is_grasped
            + 10.0 * is_success
            - 0.01 * np.sum(np.square(action))
            - 0.01
        )
        return reward

    def _check_done(self, object_pos):
        in_x = 0.28 <= object_pos[0] <= 0.52
        in_y = -0.12 <= object_pos[1] <= 0.12
        in_z = object_pos[2] >= -0.02
        if in_x and in_y and in_z:
            return True

        return self.step_count >= self.max_steps

    def render(self):
        mujoco.viewer.launch(self.model, self.data)
