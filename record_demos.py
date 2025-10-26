import os
from pathlib import Path
import pickle
import time, gc

from rlbench.environment import Environment
from rlbench.tasks import PushButton
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig


def main():
    dataset_root = os.path.expanduser("~/my_demos")
    os.makedirs(dataset_root, exist_ok=True)

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(False)
    obs_config.gripper_matrix = True
    obs_config.gripper_pose = True
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True

    #  ban cloud point
    for cam in [
        obs_config.front_camera,
        obs_config.left_shoulder_camera,
        obs_config.right_shoulder_camera,
        obs_config.overhead_camera,
        obs_config.wrist_camera,
    ]:
        cam.point_cloud = False

    action_mode = ActionMode(EndEffectorPoseViaPlanning(), Discrete())
    env = Environment(action_mode, dataset_root=dataset_root,
                      obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(PushButton)

    save_root = Path(dataset_root) / "push_button" / "variation0" / "episodes"
    total_demos = 200


    for i in range(total_demos):
        # record 1 demo
        demos = task.get_demos(amount=1, live_demos=True)
        demo = demos[0]

        episode_dir = save_root / f"episode{i}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(str(episode_dir / "low_dim_obs.pkl"), "wb") as f:
            pickle.dump(demo, f)
        print(f"Saved demo {i} at {episode_dir}/low_dim_obs.pkl")

        # clear
        gc.collect()
        time.sleep(0.1)

    env.shutdown()


if __name__ == "__main__":
    main()