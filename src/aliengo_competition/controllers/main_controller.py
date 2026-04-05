from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState, RobotState, VelocityCommand


class SpeedController(ABC):
    """Upper-level controller API: compute only speed commands."""

    def on_reset(self, state: RobotState) -> None:
        _ = state

    @abstractmethod
    def compute_command(self, state: RobotState) -> VelocityCommand:
        raise NotImplementedError


class FixedSpeedController(SpeedController):
    def __init__(self, vx: float = 0.0, vy: float = 0.0, vw: float = 0.0):
        self._command = VelocityCommand(vx=float(vx), vy=float(vy), vw=float(vw))

    def compute_command(self, state: RobotState) -> VelocityCommand:
        _ = state
        return self._command


class _CameraRenderer:
    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (RGB + Depth)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Camera rendering disabled: failed to import cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active:
            return
        rgb = camera.rgb
        depth = camera.depth
        if rgb is None or depth is None:
            return

        rgb = np.asarray(rgb)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]

        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (255.0 * depth_m / self.depth_max_m).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False


def run(
    robot: AliengoRobotInterface,
    controller: SpeedController,
    steps: int = 1000,
    render_camera: bool = False,
    camera_depth_max_m: float = 10.0,
    reset_on_fall: bool = True,
) -> None:
    total_steps = max(int(steps), 1)
    renderer = _CameraRenderer(enabled=render_camera, depth_max_m=camera_depth_max_m)

    state = robot.reset()
    controller.on_reset(state)
    print(f"[Controller] dt={robot.get_control_dt():.4f}s, steps={total_steps}")

    try:
        for _ in range(total_steps):
            state = robot.read_state()
            command = controller.compute_command(state)
            robot.set_velocity_command(vx=command.vx, vy=command.vy, vw=command.vw)

            state = robot.step()
            renderer.show(state.camera)

            if reset_on_fall and robot.is_fallen():
                robot.stop()
                state = robot.reset()
                controller.on_reset(state)
    finally:
        renderer.close()
        robot.stop()


def run_fixed_speed(
    robot: AliengoRobotInterface,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    vw: float = 0.0,
    steps: int = 1000,
    render_camera: bool = False,
    camera_depth_max_m: float = 10.0,
    reset_on_fall: bool = True,
) -> None:
    run(
        robot=robot,
        controller=FixedSpeedController(vx=vx, vy=vy, vw=vw),
        steps=steps,
        render_camera=render_camera,
        camera_depth_max_m=camera_depth_max_m,
        reset_on_fall=reset_on_fall,
    )
