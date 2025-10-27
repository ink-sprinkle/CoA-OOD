import numpy as np
from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition, ConditionSet, Condition

# = Parameters (adjust as needed) =
ATTACH_DIST = 0.06         # proximity threshold for grasp triggering (for print/debug only, no physical attach)
DROP_DIST = 0.03           # threshold for drop proximity (reserved for future extension)
BUTTON_CLEARANCE = 0.03    # initial height offset when the cup is placed above the button (slightly larger for stability)
EXTRA_CUPS_N_MAX = 2       # maximum number of extra distractor cups

# threshold to determine whether the cup has been moved away (distance between button and cup centers on XY plane)
CLEAR_RADIUS = 0.12        # 12 cm, adjustable depending on scene
# lock the button joint (reset to 0) before the cup is cleared to prevent pressing
LOCK_BUTTON_BEFORE_CLEAR = True

colors = [
    ('maroon', (0.5, 0.0, 0.0)), ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)), ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)), ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)), ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)), ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)), ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)), ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)), ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)), ('white', (1.0, 1.0, 1.0)),
]

#  Custom condition: check if the cup has been moved away from the button 
class CupClearedCondition(Condition):
    def __init__(self, task_ref: 'PushButton'):
        super().__init__()
        self._task = task_ref

    def condition_met(self):
        ok = self._task._cup_cleared()
        # Return (is_satisfied, is_final_condition_part)
        return ok, True


class PushButton(Task):

    # helpers 
    def _find_gripper_tip(self) -> Dummy:
        for name in ['right_gripper_tip','left_gripper_tip','gripper_tip','panda_hand_tip','end_effector']:
            try: return Dummy(name)
            except Exception: continue
        return None

    def _table_top_z(self) -> float:
        return self.table.get_bounding_box()[5]

    def _sample_safe_xy(self):
        key = np.random.choice(list(self._safe_regions.keys()))
        x0, x1, y0, y1 = self._safe_regions[key]
        return np.random.uniform(x0, x1), np.random.uniform(y0, y1)

    def _place_on_table(self, obj: Shape, x: float, y: float):
        bb = obj.get_bounding_box()
        z = self._table_top_z() - bb[2] + 0.001
        obj.set_position([x, y, z])

    def _randomize_blocking_cup_color(self):
        if self.blocking_cup_vis is None: return
        _, rgb = colors[np.random.choice(len(colors))]
        try: self.blocking_cup_vis.set_color(rgb)
        except Exception as e:
            print("[PushButton] color Cup_blockvisible failed:", e)

    def _snap_cup_over_button(self, clearance=BUTTON_CLEARANCE):
        """Place the blocking cup directly above the button top surface."""
        cup = self.blocking_cup
        top = self.target_topPlate
        top_pos = np.array(top.get_position()); top_bb = top.get_bounding_box()
        surf_z = top_pos[2] + top_bb[5]
        bb = cup.get_bounding_box()
        off_x = 0.5*(bb[0]+bb[3]); off_y = 0.5*(bb[1]+bb[4]); off_z = bb[2]
        cup.set_position([top_pos[0]-off_x, top_pos[1]-off_y, (surf_z+clearance)-off_z])

    def _spawn_extra_cups(self, n: int):
        """Spawn n extra cups as visual distractors in safe table regions."""
        if self.cup_template is None: return
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
                   # vis.set_collidable(False)   # visual only, no collision
                   # vis.set_respondable(False)
                except Exception as e:
                    print(f"[PushButton] warn: copy Cup1_visible failed: {e}")
            self.spawned_cups.append(cup)

    def _place_waypoint3_on_table(self):
        """
        Randomly place waypoint3 in an empty region on the table,
        slightly away from the button (e.g., 12–18 cm),
        at a height of 10–15 cm above the table.SSS
        """
        btn_pos = np.array(self.target_topPlate.get_position())
        tpos = np.array(self.table.get_position())
        tbb  = self.table.get_bounding_box()

        # Table boundaries (reserve a 5 cm margin)
        x_min = tpos[0] + tbb[0] + 0.05
        x_max = tpos[0] + tbb[3] - 0.05
        y_min = tpos[1] + tbb[1] + 0.05
        y_max = tpos[1] + tbb[4] - 0.05

        # Randomly sample a point at least 0.12 m away from the button
        for _ in range(50):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            if np.linalg.norm([x - btn_pos[0], y - btn_pos[1]]) > 0.12:
                break

        # Height: 0.15 m above the table
        z = float(self._table_top_z() + 0.15)

        # Orientation: same as button pr
