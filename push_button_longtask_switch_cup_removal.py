# -*- coding: utf-8 -*-
import os
import gc
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable

# === RLBench / PyRep ===
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig


# ===== Parameters =====
BUTTON_CLEARANCE = 0.03    # clearance height when the cup is "snapped" above the button
EXTRA_CUPS_N_MAX = 2       # maximum number of extra distracting cups
CLEAR_RADIUS = 0.12        # threshold radius for “cup cleared” (distance between button and cup centers on XY plane)
ATTACH_DIST = 0.06         # for debugging only: proximity hint for grasping
COLLISION_MARGIN = 0.05    # safety margin from table boundaries
PREPOSE_Z = 0.15           # waypoint3 height = 0.15 m above the table

colors = [
    ('maroon',  (0.5, 0.0, 0.0)), ('green', (0.0, 0.5, 0.0)),
    ('blue',    (0.0, 0.0, 1.0)), ('navy',  (0.0, 0.0, 0.5)),
    ('yellow',  (1.0, 1.0, 0.0)), ('cyan',  (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)), ('silver',(0.75, 0.75, 0.75)),
    ('gray',    (0.5, 0.5, 0.5)), ('orange',(1.0, 0.5, 0.0)),
    ('olive',   (0.5, 0.5, 0.0)), ('purple',(0.5, 0.0, 0.5)),
    ('teal',    (0.0, 0.5, 0.5)), ('azure', (0.0, 0.5, 1.0)),
    ('violet',  (0.5, 0.0, 1.0)), ('rose',  (1.0, 0.0, 0.5)),
    ('black',   (0.0, 0.0, 0.0)), ('white', (1.0, 1.0, 1.0)),
]


# ===== Stable condition (set hold_steps=1 for “immediate success”) =====
class StableCondition(Condition):
    def __init__(self, pred: Callable[[], bool], hold_steps: int = 1):
        super().__init__()
        self._pred = pred
        self._hold = int(max(1, hold_steps))
        self._cnt = 0

    def condition_met(self) -> Tuple[bool, bool]:
        ok = bool(self._pred())
        self._cnt = self._cnt + 1 if ok else 0
        reached = self._cnt >= self._hold
        return reached, True   # True = terminal condition

class PushButton(Task):
    """Record only the remove part: success when the cup is moved away from the button; no pressing phase."""

    # RLBench passes (pyrep, robot)
    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot)

    # ---------- helpers ----------
    def _find_gripper_tip(self) -> Optional[Dummy]:
        for name in ['right_gripper_tip','left_gripper_tip','gripper_tip','panda_hand_tip','end_effector']:
            try:
                return Dummy(name)
            except Exception:
                continue
        return None

    def _table_top_z(self) -> float:
        return self.table.get_bounding_box()[5]

    def _sample_safe_xy(self) -> Tuple[float, float]:
        key = np.random.choice(list(self._safe_regions.keys()))
        x0, x1, y0, y1 = self._safe_regions[key]
        return float(np.random.uniform(x0, x1)), float(np.random.uniform(y0, y1))

    def _place_on_table(self, obj: Shape, x: float, y: float):
        bb = obj.get_bounding_box()
        z = self._table_top_z() - bb[2] + 0.001
        obj.set_position([x, y, z])

    def _randomize_blocking_cup_color(self):
        if self.blocking_cup_vis is None:
            return
        _, rgb = colors[np.random.choice(len(colors))]
        try:
            self.blocking_cup_vis.set_color(rgb)
        except Exception as e:
            print("[Remove] color Cup_blockvisible failed:", e)

    def _snap_cup_over_button(self, clearance=BUTTON_CLEARANCE):
        """Place the blocking cup directly over the button top surface."""
        cup = self.blocking_cup
        top = self.target_topPlate
        top_pos = np.array(top.get_position()); top_bb = top.get_bounding_box()
        surf_z = top_pos[2] + top_bb[5]
        bb = cup.get_bounding_box()
        off_x = 0.5*(bb[0]+bb[3]); off_y = 0.5*(bb[1]+bb[4]); off_z = bb[2]
        cup.set_position([top_pos[0]-off_x, top_pos[1]-off_y, (surf_z+clearance)-off_z])

    def _spawn_extra_cups(self, n: int):
        """Spawn n extra visual distractor cups in safe table areas."""
        if self.cup_template is None:
            return
        for i in range(n):
            cup = self.cup_template.copy()
            cup.set_name(f"Cup_copy{i}")
            x, y = self._sample_safe_xy()
            self._place_on_table(cup, x, y)
            if self.cup_template_visual is not None:
                try:
                    vis = self.cup_template_visual.copy()
                    vis.set_name(f"Cup_copy{i}_visible")
                    vis.set_parent(cup)
                    vis.set_position([0, 0, 0], relative_to=cup)
                    _, rgb = colors[np.random.choice(len(colors))]
                    vis.set_color(rgb)
                    vis.set_collidable(False)
                    vis.set_respondable(False)
                except Exception as e:
                    print(f"[Remove] warn: copy Cup1_visible failed: {e}")
            self.spawned_cups.append(cup)

    def _place_waypoint3_on_table(self, min_dist=0.12, max_dist=0.25, h=PREPOSE_Z):
        """Randomly generate waypoint3 for the remove task (starting/placement pose)."""
        btn_pos = np.array(self.target_topPlate.get_position())
        tpos = np.array(self.table.get_position())
        tbb  = self.table.get_bounding_box()

        # Table boundaries (reserve safety margin)
        x_min = tpos[0] + tbb[0] + COLLISION_MARGIN
        x_max = tpos[0] + tbb[3] - COLLISION_MARGIN
        y_min = tpos[1] + tbb[1] + COLLISION_MARGIN
        y_max = tpos[1] + tbb[4] - COLLISION_MARGIN

        # Sample a point with distance in [min_dist, max_dist]
        for _ in range(64):
            r = float(np.random.uniform(min_dist, max_dist))
            theta = float(np.random.uniform(-np.pi, np.pi))
            x = float(btn_pos[0] + r * np.cos(theta))
            y = float(btn_pos[1] + r * np.sin(theta))
            if x_min <= x <= x_max and y_min <= y <= y_max:
                break
        z = float(self._table_top_z() + h)

        # Orientation: default (avoid referencing non-existent wp4)
        ori = [np.pi, 0.0, np.pi]

        self.wp_drop.set_position([x, y, z])
        self.wp_drop.set_orientation(ori)

    # Geometry checks 
    def _button_xy(self) -> Tuple[float, float]:
        p = np.array(self.target_button.get_position())
        return float(p[0]), float(p[1])

    def _cup_xy(self) -> Tuple[float, float]:
        p = np.array(self.blocking_cup.get_position())
        return float(p[0]), float(p[1])
    
    #Gripper state check
    def _gripper_open(self, thresh: float = 0.8) -> bool:
        """Return whether the gripper is sufficiently open; default threshold = 0.8 (range 0–1)."""
        g = getattr(self.robot, 'gripper', None)
        if g is None:
            return True  # If not found, don't block success (can also set to False)
        try:
            amt = float(g.get_open_amount())  # supported by most grippers
        except Exception:
            try:
                amt = float(g.get_opened_amount())  # some implementations use this name
            except Exception:
                # Fallback: estimate open ratio from joint positions
                try:
                    joints = getattr(g, 'joints', [])
                    rng = 0.0
                    cur = 0.0
                    for j in joints:
                        lo, hi = j.get_joint_interval()
                        rng += (hi - lo)
                        cur += (j.get_joint_position() - lo)
                    amt = cur / max(1e-6, rng)
                except Exception:
                    amt = 1.0
        return amt >= thresh


    def _cup_cleared(self) -> bool:
        bx, by = self._button_xy()
        cx, cy = self._cup_xy()
        d = np.linalg.norm([cx - bx, cy - by])
        return bool(d > CLEAR_RADIUS)
        

    # First clear, then release
    def _cleared_and_released(self) -> bool:
        # Once “cleared” occurs, remember it; then require the gripper to be open
        if not hasattr(self, "_cleared_latch"):
            self._cleared_latch = False
        if self._cup_cleared():
            self._cleared_latch = True
        return self._cleared_latch and self._gripper_open()

    def init_task(self):
        # Key objects
        self.target_button   = Shape('push_button_target')
        self.target_topPlate = Shape('target_button_topPlate')
        self.target_wrap     = Shape('target_button_wrap')
        self.joint           = Joint('target_button_joint')

        self.gripper_tip = self._find_gripper_tip()

        self.blocking_cup = Shape('Cup_block')
        try:
            self.blocking_cup_vis = Shape('Cup_blockvisible')
        except Exception:
            self.blocking_cup_vis = None
            
        self.register_graspable_objects([self.blocking_cup])
        self.wp_pick0 = Dummy('waypoint0')
        self.wp_pick1 = Dummy('waypoint1')
        self.wp_pick2 = Dummy('waypoint2')
        self.wp_drop  = Dummy('waypoint3')

        self.table = Shape('diningTable')
        try:
            self.cup_template = Shape('Cup1')
            self.cup_template_visual = Shape('Cup1_visible')
        except Exception:
            self.cup_template = None
            self.cup_template_visual = None
            print("[Remove] Cup1 or Cup1_visible not found")

        self.spawned_cups: List[Shape] = []
        self._safe_regions = {
            "left":  (-0.08, -0.01, -0.50,  0.50),
            "right": ( 0.51,  0.68, -0.50,  0.50),
            "front": (-0.01,  0.51, -0.50, -0.39),
            "back":  (-0.01,  0.51,  0.37,  0.50),
        }

        # Success condition: success when the cup is cleared (set hold_steps=3 for instant termination)
        self.register_success_conditions([StableCondition(self._cleared_and_released, hold_steps=3)])

    def init_episode(self, index: int) -> List[str]:
    
        # Clean up copied cups
        for cup in self.spawned_cups:
            try:
                cup.remove()
            except Exception:
                pass
        self.spawned_cups = []

        # Appearance
        self.target_topPlate.set_color([1.0, 0.0, 0.0])
        self.target_wrap.set_color([1.0, 0.0, 0.0])
        cname, rgb = colors[index % len(colors)]
        self.target_button.set_color(rgb)

        # Place the blocking cup on top of the button
        self._snap_cup_over_button(clearance=BUTTON_CLEARANCE)
        self._randomize_blocking_cup_color()

        # Enable physics for the cup
        try:
            self.blocking_cup.set_dynamic(True)
            self.blocking_cup.set_collidable(True)
            self.blocking_cup.set_respondable(True)
        except Exception:
            pass
        if self.blocking_cup_vis is not None:
            try:
                self.blocking_cup_vis.set_collidable(False)
                self.blocking_cup_vis.set_respondable(False)
            except Exception:
                pass

        # Optional extra distractor cups
        self._spawn_extra_cups(np.random.randint(0, EXTRA_CUPS_N_MAX + 1))
        
        # Reset
        self._cleared_latch = False  

        # Place the remove starting pose (waypoint3)
        self._place_waypoint3_on_table()

        return [f"remove the cup blocking the {cname} button"]

    def variation_count(self) -> int:
        return len(colors)

    def step(self) -> None:
        # Debug: near waypoint1
        try:
            tip = self.gripper_tip or Dummy('right_gripper_tip')
            p_tip = np.array(tip.get_position())
            p_w1  = np.array(self.wp_pick1.get_position())
            if np.linalg.norm(p_tip - p_w1) < ATTACH_DIST:
                print("[Remove] near waypoint1: cup is dynamic; close gripper to grasp.")
        except Exception:
            pass

        # Pure remove: lock the button joint to prevent accidental pressing
        try:
            self.joint.set_control_loop_enabled(True)
            self.joint.set_motor_locked_at_zero_velocity(True)
            self.joint.set_joint_target_position(0.0)
        except Exception:
            pass
