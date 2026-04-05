from __future__ import annotations

from aliengo_competition.controllers.main_controller import SpeedController
from aliengo_competition.robot_interface.types import RobotState, VelocityCommand


class UserSpeedController(SpeedController):
    """Participant-editable upper-level controller."""

    def on_reset(self, state: RobotState) -> None:
        _ = state

    def compute_command(self, state: RobotState) -> VelocityCommand:
        # Available data:
        # - state.joints.positions / state.joints.velocities
        # - state.imu.angular_velocity_xyz
        # - state.imu.linear_acceleration_xyz
        # - state.imu.projected_gravity_xyz
        # - state.camera.rgb / state.camera.depth

        # ================= CONTROL LOOP START =================
        # Replace this block with your logic.
        yaw_rate = float(state.imu.angular_velocity_xyz[2])
        vx = 0.0
        vy = 0.0
        vw = -0.5
        # ================== CONTROL LOOP END ==================

        return VelocityCommand(vx=vx, vy=vy, vw=vw)
