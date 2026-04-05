from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aliengo_competition.common.helpers import get_args
from aliengo_competition.controllers.main_controller import run
from aliengo_competition.robot_interface.factory import make_robot_interface
from scripts.controller_user import UserSpeedController

DEFAULT_CONTROLLER_RUN = (
    PROJECT_ROOT / "runs" / "gait-conditioned-agility" / "2026-04-01" / "train" / "124256.867761"
)


def main() -> None:
    args = get_args()
    if args.load_run in (None, "", -1, "-1"):
        args.load_run = str(DEFAULT_CONTROLLER_RUN)
        print(f"[Controller] Using default policy run: {args.load_run}")
    robot = make_robot_interface(
        args=args,
        task=args.task,
        mode=args.mode,
        headless=args.headless,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    run(
        robot=robot,
        controller=UserSpeedController(),
        steps=args.steps,
        render_camera=args.render_camera,
        camera_depth_max_m=args.camera_depth_max_m,
    )


if __name__ == "__main__":
    main()
