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
    task=tasj_name \
    num_train_steps=10000 \
    batch_size=32 \
    demos=50 \
    eval_every_steps=1000 \
    vis_every_steps=1000 \
    save_every_steps=1000 \
    num_eval_episodes=12
```
## Demos Recording
The file [`record_demos.py`](./record_demos.py) contains the code used for recording demonstrations.  
Add this file to the `src` folder and run it to record demonstrations for each task.  
Adjust the number of required demos as needed:
```python
total_demos = 200  # change quantity if needed
```

### Push_button task change
For each sub-experiment, replace the default RLBench task definition file `~/your_env_for_RLbench_setup/push_button.py` with one of the following:
- [`push_button_with_cups_no_block.py`](./push_button_with_cups_no_block.py): Used for OOD (cup) experiment; retrain and re-evaluate push_button task (with cups, no block).
- [`push_button_with_cups_1_block_long_task.py`](./push_button_with_cups_1_block_long_task.py): Used for the cup_removal_then_push long-horizon experiment.
- [`push_button_longtask_switch_cup_removal.py`](./push_button_longtask_switch_cup_removal.py): Used for the push_button_subtask_remove_cup switch-evaluation experiment.
- [`push_button_with_cups_no_block_random_initial_position.py`](./push_button_with_cups_no_block_random_initial_position.py): Used for the push_button_subtask_push_button random initial position experiment.

### Task Enivirronment File
Replace the default task model file `~/your_env_for_RLbench_setup/push_button.ttm` with the corresponding environment setup:
- [`push_button_with_cups_no_block.ttm`](./push_button_with_cups_no_block.ttm) For OOD (cup) experiment and random/identical initial positions.
- [`push_button_long_task.ttm`](./push_button_long_task.ttm) For cup_removal_then_push and sensitivity analysis.
- [`push_button_remove.ttm`](./push_button_remove.ttm) For push_button_subtask_remove_cup experiments.

### Demos Used in Each Experiment
- each evaluation: total_demos = 50
- training `push_button_with_cups_no_block` total_demos = 100
- training `push_button_with_cups_1_block_long_task.py` total_demos = 600
- training `push_button_longtask_switch_cup_removal` total_demos = 100
- sensitivity analysis long task evaluation total_demos = 200
Reference demonstrations are available on [`HuggingFace`]().

## OOD (Cube) Evaluation

### Code update
The file [`ood_workspace.py`](./ood_workspace.py) contains the additional code used for Cube-OOD testing.  
To enable obstacle injection during evaluation,  
Copy or update the code in this file into `src/workspace.py` of the original CoA repository  
(inside the `class WorkSpace` definition).  

If needed, adjust the obstacle color in the following line:  
```python
obs.set_color([1.0, 0.2, 0.2])  # Change color if necessary
```

### YAML update
The file [`launch_added.yaml`](./launch_added.yaml) contains the additional parameters required for Cube-OOD testing (also for all the experiments below).
Copy the contents of this file into src/cfgs/launch.yaml in the original CoA repository.
If necessary, adjust the obstacle size in the configuration:
``` python
obstacle_size: [0.05, 0.05, 0.10]  # Change size if necessary
```
### Evaluation
Use
```bash
python -m scripts.eval task=push_button snapshot=root_to_weight_clean_train.pt +obstacle_test=true
```

## OOD (Cup) Evaluation
Replace the demos in the folder`data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
Then run:
```bash
python -m scripts.eval task=push_button snapshot=root_to_weight_clean_train.pt
```

## Approach 1: Input Perturbation
The file [`input_perturbation.py`](./input_perturbation.py) contains the additional code used for input perturbation training and cube ood evaluation.  

Copy the code in this file into `src/dataset/rlbrnch_dataset.py` of the original CoA repository  
Then run the following command for training:
```python
python -m scripts.train task=tasj_name num_train_steps=10000 batch_size=32 demos=50 eval_every_steps=1000 vis_every_steps=1000 save_every_steps=1000 num_eval_episodes=12 use_augmentation = true
```
After training, evaluate the model under Cube-OOD conditions:
```bash
python -m scripts.eval task=push_button snapshot=root_to_weight_input_perturbation_train.pt +obstacle_test=true
```

## Approach 2: OOD to ID transfer
### 
