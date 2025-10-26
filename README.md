# CoA-OOD
Chain-of-action with ood test and approach to improvement

## Experimental Environment
The project was reproduced locally following the setup instructions from the official [Chain-of-Action repository](https://github.com/ByteDance-Seed/Chain-of-Action).  
The actual hardware and software configurations used in this experiment are listed below.

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

## OOD (Cube) Evaluation

### Code update
The file [`ood_workspace.py`](./ood_workspace.py) contains the additional code used for Cube-OOD testing.  
To enable obstacle injection during evaluation,  
copy or update the code in this file into `src/workspace.py` of the original CoA repository  
(inside the `class WorkSpace` definition).  

If needed, adjust the obstacle color in the following line:  
```python
obs.set_color([1.0, 0.2, 0.2])  # Change color if necessary

###YAML update
The file [`launch_added.yaml`](./launch_added.yaml) contains the additional parameters required for Cube-OOD testing (also for all the test below).
Copy the contents of this file into src/cfgs/launch.yaml in the original CoA repository.
If necessary, adjust the obstacle size in the configuration:
obstacle_size: [0.05, 0.05, 0.10]  # Change size if necessary


