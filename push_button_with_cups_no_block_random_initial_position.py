import os
import numpy as np
from math import pi
from typing import List

from pyrep.errors import IKError
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy

from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint
from rlbench.backend.conditions import JointCondition, ConditionSet

# ---- sampling parameters ----
R_MIN = 0.12
R_MAX = 0.25
Z_OFFSET = 0.20        # height offset (+15 cm)

colors = [
    ('maroon', (0.5, 0.0, 0.0)), ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),   ('navy',  (0.0, 0.0, 0.5)),
    ('yellow',(1.0, 1.0, 0.0)),  ('cyan',  (0.0, 1.0, 1.0)),
    ('magenta',(1.0, 0.0, 1.0)), ('silver',(0.75, 0.75, 0.75)),
    ('gray',  (0.5, 0.5, 0.5)),  ('orange',(1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),  ('purple',(0.5, 0.0, 0.5)),
    ('teal',  (0.0, 0.5, 0.5)),  ('azure', (0.0, 0.5, 1.0)),
    ('violet',(0.5, 0.0, 1.0)),  ('rose',  (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),  ('white', (1.0, 1.0, 1.0)),
]

class PushButton(Task):

    def init_task(self):
        self.target_button = Shape('push_button_target')
        self.target_topPlate = Shape('target_button_topPlate')
        self.joint = Joint('target_button_joint')
        self.target_wrap = Shape('target_button_wrap')
        self.goal_condition = JointCondition(self.joint, 0.003)
        

        # Robot arm handle & home pose
        self.arm = self.robot.arm
        self._home_q = self.arm.get_joint_positions()
        self._episode_start_q = None        # randomized starting joint configuration of this episode
        self._episode_start_pos_w = None    # starting pose in world coordinates (for logging only)

        # (Important) Remove forced waypoint0 reset. Demonstrations will overwrite waypoints,
        # and this reset can interfere. If forced initialization is needed before demos,
        # it can be restored manually, but training should initialize in init_episode() instead.

        # Cup templates
        try:
            self.cup_template = Shape('Cup1')
            self.cup_template_visual = Shape('Cup1_visible')
        except Exception:
            self.cup_template = None
            self.cup_template_visual = None
            print("[PushButton] Cup1 or Cup1_visible not found")

        self.spawned_cups = []

    def get_base(self) -> Dummy:
        if getattr(self, "_base_object", None) is not None and self._base_object.still_exists():
            return self._base_object
        if Object.exists(self.name):
            self._base_object = Dummy(self.name)
            return self._base_object
        ttm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../task_ttms/%s.ttm" % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(f"[PushButton] TTM not found: {ttm_file}")
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object

    # Before reachability validation, pre-randomize the starting pose (fallback);
    # the final effective pose is still determined at the end of init_episode().
    def validate(self):
        self._ensure_random_start_pose()
        super().validate()

    def init_episode(self, index: int) -> List[str]:
        # Clean up previously spawned cups
        for cup in self.spawned_cups:
            try:
                cup.remove()
            except Exception:
                pass
        self.spawned_cups = []

        # Appearance setup
        self.target_topPlate.set_color([1.0, 0.0, 0.0])
        self.target_wrap.set_color([1.0, 0.0, 0.0])
        name, rgb = colors[index % len(colors)]
        self.target_button.set_color(rgb)
        self.register_success_conditions([ConditionSet([self.goal_condition], True, False)])

        # (Optional fallback) Ensure randomized start pose again
        self._ensure_random_start_pose()

        # Randomly generate 1–3 cups
        if self.cup_template is not None:
            n_extra = np.random.randint(1, 4)
            print(f"[PushButton] this episode generate {n_extra} cup(s)")
            for i in range(n_extra):
                cup = self.cup_template.copy()
                cup.set_name(f"Cup_copy{i}")
                pos = self._sample_free_area_position(cup)
                cup.set_position(pos)

                cup_vis = None
                if self.cup_template_visual is not None:
                    try:
                        cup_vis = self.cup_template_visual.copy()
                        cup_vis.set_name(f"Cup_copy{i}_visible")
                        cup_vis.set_parent(cup)
                        cup_vis.set_position([0, 0, 0], relative_to=cup)
                    except Exception as e:
                        print(f"[PushButton] warning: failed to copy Cup1_visible: {e}")
                if cup_vis is not None:
                    try:
                        _, rgb2 = colors[np.random.choice(len(colors))]
                        cup_vis.set_color(rgb2)
                    except Exception as e:
                        print(f"[PushButton] color failed: {e}")

                self.spawned_cups.append(cup)

        #Place randomization at the end of episode initialization
        # and synchronize the joint target to avoid pull-back by the controller
        ok = self._randomize_arm_start_pose(max_retries=20)
        if not ok:
            # Fallback to home; set target = home as well to prevent controller drift
            try:
                self.arm.set_joint_positions(self._home_q, disable_dynamics=True)
            except TypeError:
                self.arm.set_joint_positions(self._home_q)
            self._episode_start_q = self._home_q
            self._episode_start_pos_w = None
            try:
                self.arm.set_joint_target_positions(self._episode_start_q)
            except AttributeError:
                pass
        else:
            # On successful randomization, set target = same joint angles to avoid control pull-back
            try:
                self.arm.set_joint_target_positions(self._episode_start_q)
            except AttributeError:
                pass

        # Print this episode’s starting pose (world coordinates + joint angles)
        if self._episode_start_q is not None:
            p = None if self._episode_start_pos_w is None else np.round(self._episode_start_pos_w, 3)
            q = np.round(self._episode_start_q, 3)
            print(f"[PushButton] episode start pose: world_pos={p}, q={q.tolist()}")

        return [
            f'push the {name} button',
            f'push down the {name} button',
            f'press the button with the {name} base',
            f'press the {name} button'
        ]

    def variation_count(self) -> int:
        return len(colors)

    def step(self) -> None:
        if self.goal_condition.condition_met() == (True, True):
            self.target_topPlate.set_color([0.0, 1.0, 0.0])
            self.target_wrap.set_color([0.0, 1.0, 0.0])

    def cleanup(self) -> None:
        # Reset to home; clear episode caches and remove spawned cups
        try:
            self.arm.set_joint_positions(self._home_q, disable_dynamics=True)
        except TypeError:
            self.arm.set_joint_positions(self._home_q)
        self._episode_start_q = None
        self._episode_start_pos_w = None
        for cup in self.spawned_cups:
            try:
                cup.remove()
            except Exception:
                pass
        self.spawned_cups = []

    def _ensure_random_start_pose(self):
        """If not yet randomized, perform one randomization; fallback to home on failure."""
        if self._episode_start_q is not None:
            return
        ok = self._randomize_arm_start_pose()
        if not ok:
            try:
                self.arm.set_joint_positions(self._home_q, disable_dynamics=True)
            except TypeError:
                self.arm.set_joint_positions(self._home_q)
            self._episode_start_q = self._home_q
            self._episode_start_pos_w = None

    def _sample_free_area_position(self, cup: Shape):
        """Sample a safe area on the table to place a cup."""
        table = Shape('diningTable')
        bbox = table.get_bounding_box()
        table_z = bbox[5]
        candidates = {
            "left":  (-0.08, -0.01, -0.50, 0.50),
            "right": (0.51, 0.68, -0.50, 0.50),
            "front": (-0.01, 0.51, -0.50, -0.39),
            "back":  (-0.01, 0.51,  0.37,  0.50),
        }
        region = np.random.choice(list(candidates.keys()))
        x0, x1, y0, y1 = candidates[region]
        x = np.random.uniform(x0, x1)
        y = np.random.uniform(y0, y1)
        z_min = cup.get_bounding_box()[2]
        z = table_z - z_min + 0.001
        print(f"[PushButton] set cup at {region} region: ({x:.3f}, {y:.3f}, {z:.3f})")
        return [x, y, z]

    def _randomize_arm_start_pose(self, max_retries: int = 80) -> bool:
        """Randomize a reachable starting pose around the button
        within radius 12–25 cm and Z offset +15 cm, with downward end-effector orientation."""
        # Define orientation range relative to button→base direction for more natural poses
        tip_xy = np.array(self.arm.get_tip().get_position()[:2])
        btn_xy = np.array(self.target_button.get_position()[:2])
        vec = tip_xy - btn_xy
        facing_yaw = np.arctan2(vec[1], vec[0])
        half_span = np.deg2rad(120.0)
        theta_min = facing_yaw - half_span
        theta_max = facing_yaw + half_span

        # End-effector “facing down” Euler angles:
        # align Z-axis toward ground (common grasping orientation)
        # Use (roll, pitch, yaw) = (pi, 0, yaw); for other robots, may adjust to (0, pi, yaw)
        def down_euler_candidates(yaw: float):
            y = float(yaw)
            return [
                [np.pi, 0.0, y],        # Z-down: roll=pi
                [0.0, np.pi, y],        # Z-down: pitch=pi
                [np.pi, 0.0, y + np.pi/2.0],
                [0.0, np.pi, y + np.pi/2.0],
                [np.pi, 0.0, y + np.pi],   # backup with 180° rotation
                [0.0, np.pi, y + np.pi],
            ]

        self._episode_start_pos_w = None

        for t in range(1, max_retries + 1):
            # 1) Sample a relative target position w.r.t. the button
            r = float(np.random.uniform(R_MIN, R_MAX))
            theta = float(np.random.uniform(theta_min, theta_max))
            rel_pos = [r * np.cos(theta), r * np.sin(theta), Z_OFFSET]

            # 2) Convert to world coordinates
            tmp = Dummy.create()
            try:
                tmp.set_position(rel_pos, relative_to=self.target_button)
                target_pos_w = tmp.get_position()
            finally:
                tmp.remove()

            # 3) Use home pose as IK seed (more stable)
            try:
                self.arm.set_joint_positions(self._home_q, disable_dynamics=True)
            except TypeError:
                self.arm.set_joint_positions(self._home_q)

            # 4) Orient end-effector toward button center, try multiple downward orientations
            yaw_world = np.arctan2(target_pos_w[1] - btn_xy[1], target_pos_w[0] - btn_xy[0])
            for euler in down_euler_candidates(yaw_world):
                try:
                    q = self.arm.solve_ik(position=target_pos_w, euler=euler)
                except IKError:
                    q = None

                if q is not None:
                    # Success: apply joint configuration and record it
                    self._episode_start_q = q
                    self._episode_start_pos_w = target_pos_w
                    try:
                        self.arm.set_joint_positions(q, disable_dynamics=True)
                    except TypeError:
                        self.arm.set_joint_positions(q)

                    p = np.round(target_pos_w, 3)
                    e = np.round(euler, 3)
                    print(f"[PushButton] start pose set (try {t}): world_pos={p}, euler={e}")
                    return True

        print("[PushButton][WARN] IK failed after retries; fallback to home.")
        return False
