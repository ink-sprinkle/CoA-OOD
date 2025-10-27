# CoA-OOD
Chain-of-action with ood test and approach to improvement

## Experimental Environment
The project was reproduced locally following the setup instructions from the official [Chain-of-Action repository](https://github.com/ByteDance-Seed/Chain-of-Action).  
The actual hardware and software configurations used in this experiment are listed below.
```bash
| Component | Specification |
GPU	          Nvidia RTX 3080 Laptop GPU (16 GB VRAM)
CPU	          AMD Ryzen 9 5900HX
Memory	      32 GB RAM
Operating System	Ubuntu 20.04.6 LTS
Python	      3.9.23 (via Miniconda3)
RLBench	      1.2.0
PyRep	        4.1.0.3
CoppeliaSim	  v4.1 EDU
PyTorch	      2.5.1 + cu121
```
## Experiment Reproduction under Limited Training

### Dataset and Tasks
Training datasets were obtained following the instructions in the official repository.  
Four RLBench tasks were selected for reproduction:
- push_button  
- pickup_cup  
- reach_target  
- press_switch

### Training Parameters
Training was performed under limited GPU memory.  
Key parameters were adjusted as follows:

```bash
python -m scripts.train \
    task=push_button \
    num_train_steps=10000 \
    batch_size=32 \
    demos=50 \
    eval_every_steps=1000 \
    vis_every_steps=1000 \
    save_every_steps=1000 \
    num_eval_episodes=12
```
## Demos Recording
The file [`recrord_demos.py`](./record_demos.py) contains the additional code used for Cube-OOD testing.  
Add this file to the folder src
And run this file, adjust the number of required demos
``` python
total_demos = 200 # change quantity if needed
```
for each task applied different RLBench task envirionment setup code

### Push_button task change
for each sub experiment replace the file `~/your_env_for_RLbench_setup/push_button.py`
- The file [`push_button_with_cups_no_block.py`](./push_button_with_cups_no_block.py) contains new task description code for OOD (cup) experiment, retrain and reevaluate push_button task (with cups no boock)
- The file [`push_button_with_cups_1_block_long_task.py`](./ppush_button_with_cups_1_block_long_task.py) contains new task description for cup_removal_then_push retrain and reevaluate experiment
- The file [`push_button_longtask_switch_cup_removal.py`](./push_button_longtask_switch_cup_removal.py) contains new task description code for push_button_subtask_remov_cup retrain and reevaluate experiment
- The file [`push_button_with_cups_no_block.py`](./push_button_with_cups_no_block.py) contains new task description code for OOD (cup) experiment, retrain and reevaluate push_button task 
- The file [`push_button_with_cups_no_block.py`](./push_button_with_cups_no_block.py) contains new task description code for OOD (cup) experiment, retrain and reevaluate push_button task (with cups no boock)
## OOD (Cube) Evaluation

### Code update
The file [`ood_workspace.py`](./ood_workspace.py) contains the additional code used for Cube-OOD testing.  
To enable obstacle injection during evaluation,  
copy or update the code in this file into `src/workspace.py` of the original CoA repository  
(inside the `class WorkSpace` definition).  

If needed, adjust the obstacle color in the following line:  
```python
obs.set_color([1.0, 0.2, 0.2])  # Change color if necessary
```

### YAML update
The file [`launch_added.yaml`](./launch_added.yaml) contains the additional parameters required for Cube-OOD testing (also for all the test below).
Copy the contents of this file into src/cfgs/launch.yaml in the original CoA repository.
If necessary, adjust the obstacle size in the configuration:
``` python
obstacle_size: [0.05, 0.05, 0.10]  # Change size if necessary
```

