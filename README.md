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
    task=task_name \
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
Reference demonstrations are available on [`HuggingFace`](https://huggingface.co/datasets/ink-sprinkle/CoA-OOD-and-improvement-experiment/tree/main).

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
Run the Cube-OOD evaluation:
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
```bash
python -m scripts.train task=tasj_name num_train_steps=10000 batch_size=32 demos=50 eval_every_steps=1000 vis_every_steps=1000 save_every_steps=1000 num_eval_episodes=12 use_augmentation = true
```
After training, evaluate the model under Cube-OOD conditions:
```bash
python -m scripts.eval task=push_button snapshot=root_to_weight_input_perturbation_train.pt +obstacle_test=true
```

## Approach 2: OOD to ID transfer
### Cups in Environment no Block
Replace the demos in the folder `data/rlbench/train/push_button/variation0` and `data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
Replace the task description as 'Task Enivirronment File' said.
Then run:
```bash
python -m scripts.train task=push_button num_train_steps=10000 batch_size=32 demos=50 eval_every_steps=1000 vis_every_steps=1000 save_every_steps=1000 num_eval_episodes=12
```

### Long Task (Remove then Push)
The file [`fixed_action_sequence.py`](./fixed_action_sequence.py) contains the additional code used for long-horizon task training.  
To enable fixed action sequence loading during training,  
copy or update the code in this file into  
`src/rlbench_env.py` of the original CoA repository  
(inside the class `RLBenchEnvFactory(EnvFactory)`,  
under the function `_load_demos(self, cfg, training=True):`).

This update fixes the `action_sequence` length to the maximum value among recorded episodes,  
ensuring stable training for long-horizon experiments under limited computational resources  
(the same method is also applied for switch training).

To adjust the action sequence length, modify the value in  
`src/cfgs/base/base_configs.yaml`:

```python
action_sequence: 190  # ActionSequenceWrapper
```
Replace the demos in the folder `data/rlbench/train/push_button/variation0` and `data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
Replace the task description as 'Task Enivirronment File' said.
Then run the following command for training 3 times:
```bash
python -m scripts.train task=tasj_name num_train_steps=40000 batch_size=32 demos=200 eval_every_steps=2000 vis_every_steps=1000 save_every_steps=2000 num_eval_episodes=12
```

### Switch for removing training
Replace the demos in the folder `data/rlbench/train/push_button/variation0` and `data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
Replace the task description as 'Task Enivirronment File' said.
Then run the following command for training:
```bash
python -m scripts.train task=tasj_name num_train_steps=10000 batch_size=32 demos=100 eval_every_steps=1000 vis_every_steps=1000 save_every_steps=1000 num_eval_episodes=12
```

### Switch for pushing training
Replace the demos in the folder `data/rlbench/train/push_button/variation0` and `data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
Replace the task description as 'Task Enivirronment File' said.
Then run the following command for training:
```bash
python -m scripts.train task=tasj_name num_train_steps=20000 batch_size=32 demos=200 eval_every_steps=2000 vis_every_steps=1000 save_every_steps=2000 num_eval_episodes=12
```
### Switching Evaluation
Put the highest success rate for each sub branch training checkpoint in to the folder `srcipts/checkpoint` as `remove.pt` and `push.pt`
Put the file [`eval_switch.py`](./eval.switch.py) into the folder `srripts`
Put the file [`router_workspace.py`](./roiter_workspace.py) into the folder `src`
Replace the demos in the folder`data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
And run
```bash
python -m scripts.eval_switch
```
## Sensitivity analysis
Replace the demos in the folder and `data/rlbench/eval/push_button/variation0` with the generated (or downloaded) demos.
And run
```bash
python -m scripts.eval task=push_button snapshot=root_to_weight_long_task_train.pt
```
