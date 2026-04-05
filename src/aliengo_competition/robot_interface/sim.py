from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import (
    CameraState,
    ImuState,
    JointState,
    RobotState,
    VelocityCommand,
)


@dataclass
class StepResult:
    observation: Any
    reward: torch.Tensor | None
    done: torch.Tensor | None
    info: Dict[str, Any]


class SimAliengoRobot(AliengoRobotInterface):
    CMD_VX = 0
    CMD_VY = 1
    CMD_VW = 2
    CMD_BODY_PITCH = 10

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self._base_env = self._unwrap_env()

        self._dt = self._infer_dt()
        self._step_index = 0
        self._sim_time_s = 0.0
        self._command_template = None
        self._velocity_command = VelocityCommand()
        self._body_pitch = 0.0
        self._prev_base_lin_vel = None

        self._joint_names = self._extract_joint_names()

        self._last_result = StepResult(
            observation=self.env.get_observations(),
            reward=None,
            done=None,
            info={},
        )
        self._refresh_command_template()
        self._seed_imu_reference()

    def _unwrap_env(self):
        env = self.env
        while hasattr(env, "env") and getattr(env, "env") is not env:
            env = env.env
        return env

    def _infer_dt(self) -> float:
        dt = getattr(self._base_env, "dt", None)
        try:
            value = float(dt)
            if value > 0.0:
                return value
        except (TypeError, ValueError):
            pass
        return 0.02

    def _extract_joint_names(self):
        names = getattr(self._base_env, "dof_names", None)
        if names is None:
            count = int(getattr(self._base_env, "num_dof", 0))
            return tuple(f"joint_{idx}" for idx in range(count))
        return tuple(str(name) for name in names)

    def _row_numpy(self, tensor, default_dim: int):
        if tensor is None:
            return np.zeros(default_dim, dtype=np.float32)
        if torch.is_tensor(tensor):
            if tensor.ndim > 1:
                tensor = tensor[0]
            return tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        arr = np.asarray(tensor, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[0]
        return arr.copy()

    def _refresh_command_template(self) -> None:
        template = None
        default_command = getattr(self._base_env, "default_command", None)
        if torch.is_tensor(default_command):
            template = default_command.detach().clone()
        elif hasattr(self._base_env, "commands") and torch.is_tensor(self._base_env.commands):
            template = self._base_env.commands[0].detach().clone()

        if template is None:
            self._command_template = None
            return

        # Keep deterministic default gait parameters and expose only speed
        # commands (vx, vy, vw) to participant controllers.
        fixed_values = {
            3: 0.0,
            4: 3.0,
            5: 0.5,
            6: 0.0,
            7: 0.0,
            8: 0.5,
            9: 0.08,
            11: 0.0,
            12: 0.25,
            13: 0.45,
            14: 0.0,
        }
        for index, value in fixed_values.items():
            if template.shape[0] > index:
                template[index] = float(value)

        self._command_template = template

    def _apply_command(self) -> None:
        vx = float(self._velocity_command.vx)
        vy = float(self._velocity_command.vy)
        vw = float(self._velocity_command.vw)
        pitch = float(self._body_pitch)

        if hasattr(self.env, "set_command"):
            self.env.set_command(vx, vy, vw, pitch)
            return

        if self._command_template is None:
            self._refresh_command_template()
        if self._command_template is None or not hasattr(self._base_env, "commands"):
            raise AttributeError("Environment does not expose a controllable command interface.")

        command = self._command_template.clone()
        command[self.CMD_VX] = vx
        command[self.CMD_VY] = vy
        command[self.CMD_VW] = vw
        if command.shape[0] > self.CMD_BODY_PITCH:
            command[self.CMD_BODY_PITCH] = pitch
        self._base_env.commands[:] = command.unsqueeze(0).repeat(self._base_env.num_envs, 1)

    def _get_camera_state(self) -> CameraState:
        camera_getter = getattr(self._base_env, "get_front_camera_data", None)
        if not callable(camera_getter):
            return CameraState(rgb=None, depth=None)

        payload = camera_getter(env_id=0)
        if not isinstance(payload, dict):
            return CameraState(rgb=None, depth=None)
        rgb = payload.get("image")
        depth = payload.get("depth")
        return CameraState(
            rgb=None if rgb is None else np.asarray(rgb).copy(),
            depth=None if depth is None else np.asarray(depth, dtype=np.float32).copy(),
        )

    def _seed_imu_reference(self) -> None:
        self._prev_base_lin_vel = self._row_numpy(getattr(self._base_env, "base_lin_vel", None), default_dim=3)

    def _build_state(self, *, update_imu_reference: bool) -> RobotState:
        dof_pos = self._row_numpy(getattr(self._base_env, "dof_pos", None), default_dim=len(self._joint_names))
        dof_vel = self._row_numpy(getattr(self._base_env, "dof_vel", None), default_dim=len(self._joint_names))
        base_lin_vel = self._row_numpy(getattr(self._base_env, "base_lin_vel", None), default_dim=3)
        base_ang_vel = self._row_numpy(getattr(self._base_env, "base_ang_vel", None), default_dim=3)
        projected_gravity = self._row_numpy(getattr(self._base_env, "projected_gravity", None), default_dim=3)
        base_quat = self._row_numpy(getattr(self._base_env, "base_quat", None), default_dim=4)

        if self._prev_base_lin_vel is None:
            linear_accel = np.zeros(3, dtype=np.float32)
        else:
            linear_accel = (base_lin_vel - self._prev_base_lin_vel) / max(self._dt, 1e-6)
        if update_imu_reference:
            self._prev_base_lin_vel = base_lin_vel.copy()

        joints = JointState(
            names=self._joint_names,
            positions=dof_pos,
            velocities=dof_vel,
        )
        imu = ImuState(
            angular_velocity_xyz=base_ang_vel,
            linear_acceleration_xyz=linear_accel.astype(np.float32, copy=False),
            projected_gravity_xyz=projected_gravity,
            orientation_xyzw=base_quat,
        )

        return RobotState(
            step_index=int(self._step_index),
            sim_time_s=float(self._sim_time_s),
            dt=float(self._dt),
            joints=joints,
            imu=imu,
            base_linear_velocity_xyz=base_lin_vel,
            base_angular_velocity_xyz=base_ang_vel,
            camera=self._get_camera_state(),
        )

    def set_speed(self, vx: float, vy: float, vw: float) -> None:
        self.set_velocity_command(vx=vx, vy=vy, vw=vw)

    def set_velocity_command(self, vx: float, vy: float, vw: float) -> None:
        self._velocity_command = VelocityCommand(vx=float(vx), vy=float(vy), vw=float(vw))
        self._apply_command()

    def set_body_pitch(self, pitch: float) -> None:
        self._body_pitch = float(pitch)
        self._apply_command()

    def stop(self) -> None:
        self._velocity_command = VelocityCommand()
        self._body_pitch = 0.0
        self._apply_command()

    def reset(self):
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, privileged_obs = result
            info = {"privileged_obs": privileged_obs}
        else:
            obs = result
            info = {}
            if isinstance(obs, dict) and "privileged_obs" in obs:
                info["privileged_obs"] = obs["privileged_obs"]

        self._last_result = StepResult(observation=obs, reward=None, done=None, info=info)
        self._refresh_command_template()
        self.stop()
        self._step_index = 0
        self._sim_time_s = 0.0
        self._seed_imu_reference()
        return self.read_state()

    def step(self):
        obs = self.env.get_observations()
        policy_input = obs.detach() if hasattr(obs, "detach") else obs
        action = self.policy(policy_input)
        env_action = action.detach() if hasattr(action, "detach") else action
        result = self.env.step(env_action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, privileged_obs, reward, done, info = result
            info["privileged_obs"] = privileged_obs
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            if isinstance(obs, dict) and "privileged_obs" in obs:
                info["privileged_obs"] = obs["privileged_obs"]
        else:
            raise ValueError("Unexpected environment step() return format.")

        self._step_index += 1
        self._sim_time_s += self._dt
        self._last_result = StepResult(observation=obs, reward=reward, done=done, info=info)
        return self._build_state(update_imu_reference=True)

    def read_state(self) -> RobotState:
        return self._build_state(update_imu_reference=False)

    def get_camera(self):
        state = self.read_state()
        return {"image": state.camera.rgb, "depth": state.camera.depth}

    def get_observation(self):
        return self._last_result.observation

    def get_control_dt(self) -> float:
        return float(self._dt)

    def is_fallen(self) -> bool:
        if self._last_result.done is None:
            return False
        return bool(torch.any(self._last_result.done).item())
