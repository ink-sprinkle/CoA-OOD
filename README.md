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
