#!/usr/bin/env python
import os, sys, torch, hydra
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.router_workspace import RouterWorkspace
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path

@hydra.main(config_path="../src/cfgs/", config_name="launch", version_base=None)
def main(cfg):
    torch.serialization.add_safe_globals([DictConfig])

    # 使用 push 的 checkpoint 对齐 method/env 配置
    ckpt_path = to_absolute_path(cfg.router.push_snapshot)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_ckpt = ckpt["config"]

    cfg.env = cfg_ckpt.env
    cfg.method = cfg_ckpt.method
    cfg.method_name = cfg_ckpt.method_name
    cfg.action_sequence = cfg_ckpt.action_sequence
    if "dataset" in cfg_ckpt:
        cfg.dataset = cfg_ckpt.dataset
    if "eval" in cfg_ckpt:
        cfg.eval = cfg_ckpt.eval
    cfg.wandb.use = False

    # 直接运行评估
    ws = RouterWorkspace(cfg)
    ws.eval()

if __name__ == "__main__":
    main()
