import numpy as np
from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition,ConditionSet

# button top plate and wrapper will be be red before task completion
# and be changed to cyan upon success of task, so colors list used to randomly vary colors of
# base block will be redefined, excluding red and green
colors = [
    ('maroon', (0.5, 0.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]


class PushButton(Task):

    def init_task(self):
        self.target_button = Shape('push_button_target')
        self.target_topPlate = Shape('target_button_topPlate')
        self.joint = Joint('target_button_joint')
        self.target_wrap = Shape('target_button_wrap')
        self.goal_condition = JointCondition(self.joint, 0.003)
        
        # get cup from RLBench
        try: 
            self.cup_template = Shape('Cup1') 
            self.cup_template_visual = Shape('Cup1_visible')

        except Exception: 
            self.cup_template = None 
            self.cup_template_visual = None
            print("[PushButton] Cup1 or Cup1_visible not found")

        self.spawned_cups = []
        
    def _sample_free_area_position(self, cup: Shape):
        # table height 
        table = Shape('diningTable') 
        bbox = table.get_bounding_box()
        table_z = bbox[5]   # top of the table
        

        # possible rigion
        candidates = {
            "left":  (-0.08, -0.01, -0.50, 0.50),
            "right": (0.51, 0.68, -0.50, 0.50),
            "front": (-0.01, 0.51, -0.50, -0.39),
            "back":  (-0.01, 0.51,  0.37,  0.50),
        }

        # random rigion
        region = np.random.choice(list(candidates.keys()))
        x0, x1, y0, y1 = candidates[region]

        # random point
        x = np.random.uniform(x0, x1)
        y = np.random.uniform(y0, y1)
        
        # let the botton on the top of the table
        bbox = cup.get_bounding_box()
        z_min = bbox[2]
        z = table_z -z_min + 0.001
        return [x, y, z]

        print(f"[PushButton] set cup at {region} region: ({x:.3f}, {y:.3f}, {z:.3f})")
        return [x, y, z]


    def init_episode(self, index: int) -> List[str]:
    
        # cups clead=r
        for cup in self.spawned_cups:
            try:
                cup.remove()
            except Exception:
                pass
        self.spawned_cups = []
        self._variation_index = index
        self.target_topPlate.set_color([1.0, 0.0, 0.0])
        self.target_wrap.set_color([1.0, 0.0, 0.0])
        self.variation_index = index
        button_color_name, button_rgb = colors[index]
        self.target_button.set_color(button_rgb)
        self.register_success_conditions(
            [ConditionSet([self.goal_condition], True, False)])
        # ===== new：random set cup location =====
        if self.cup_template is not None:
            n_extra = np.random.randint(1, 4) #random 1-3 cups
            print(f"[PushButton] this episode generate {n_extra} cup(s)")
            for i in range(n_extra):
                cup = self.cup_template.copy()   # copy cup
                cup.set_name(f"Cup_copy{i}")
                        # calculate the cup height
               # bbox = cup.get_bounding_box()
                bbox = self.cup_template.get_bounding_box()
                print("[Debug] Cup1 bbox =", bbox)

                #obj_height = bbox[5] - bbox[2]
                
                pos = self._sample_free_area_position(cup)
                cup.set_position(pos)                
                parent = cup.get_parent()
                print(f"[Debug] {cup.get_name()} parent =", parent.get_name() if parent else "None")
                
               # bbox = cup.get_bounding_box()
                #x_min, y_min, z_min, x_max, y_max, z_max = bbox
               # size = [x_max - x_min, y_max - y_min, z_max - z_min]
                #print(f"[Debug] {cup.get_name()} bbox = {bbox}, size={size}")


                # 复制可见子节点并挂到外壳
                cup_vis = None
                if self.cup_template_visual is not None:
                    try:
                        cup_vis = self.cup_template_visual.copy()
                        cup_vis.set_name(f"Cup_copy{i}_visible")
                        cup_vis.set_parent(cup)
                        cup_vis.set_position([0, 0, 0], relative_to=cup)
                    except Exception as e:
                        print(f"[PushButton] warning: failed to copy Cup1_visible: {e}")
                else:
                    print(f"[PushButton] warning: no Cup1_visible template, cup will be invisible.")                

                # random color（only for visible）
                if cup_vis is not None:
                    print(f"[Debug] {cup_vis.get_name()} pos={cup_vis.get_position()}")
                    try:
                        _, rgb = colors[np.random.choice(len(colors))]
                        cup_vis.set_color(rgb)
                        print(f"[PushButton] colored {cup_vis.get_name()} with {rgb}")
                    except Exception as e:
                        print(f"[PushButton] color failed: {e}")

                self.spawned_cups.append(cup)
                print("cup pivot (world):", cup.get_position(relative_to=None))
                print("cup_vis pivot (world):", cup_vis.get_position(relative_to=None))
                pos1 = cup.get_position(relative_to=None)      #  pivot is in world coordinates
                bbox1 = cup.get_bounding_box()                 # geometric range relative to the pivot
                z_min_world1 = pos1[2] + bbox1[2]                # lowest point of the geometry in world coordinates (z)
                z_max_world1 = pos1[2] + bbox1[5]                # highest point of the geometry in world coordinates (z)          
                print(f"  -> {cup.get_name()} pivot={pos1}, world_z_range=[{z_min_world1:.3f}, {z_max_world1:.3f}]")
     
                #print(f"  -> cup{i}: name={cup.get_name()} pos={cup.get_position()}")      
                              
                      
        return ['push the %s button' % button_color_name,
                'push down the %s button' % button_color_name,
                'press the button with the %s base' % button_color_name,
                'press the %s button' % button_color_name]
                

    def variation_count(self) -> int:
        return len(colors)

    def step(self) -> None:
        if self.goal_condition.condition_met() == (True, True):
            self.target_topPlate.set_color([0.0, 1.0, 0.0])
            self.target_wrap.set_color([0.0, 1.0, 0.0])