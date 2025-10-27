
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from pyrep.objects.shape import Shape
from src.workspace import Workspace
from hydra.utils import to_absolute_path
import re

import hashlib

def _state_dict_fingerprint(sd) -> str:
    """给 checkpoint 生成一个简短指纹，方便日志里确认加载的是谁"""
    m = hashlib.sha256()
    # 只hash权重内容和shape，避免受字典顺序影响
    for k in sorted(sd.keys()):
        t = sd[k]
        m.update(k.encode())
        m.update(str(tuple(t.shape)).encode())
        m.update(t.cpu().numpy().tobytes())
    return m.hexdigest()[:12]  # 取前12位即可

# =========================
# 路由策略类（负责两模型切换）
# =========================
class TwoAgentRouter:
    def __init__(self, remove_agent, push_agent, button_name="push_button_target",
                 cup_names=None, clear_radius=0.12,
                 check_period=30, stop_sensing_after_push=True,
                 min_dwell_steps=0, verbose=False, z_tol=0.08, priority_z_tol=None, center_clear_radius=0.12,
                 r_btn=0.02, priority_names=None):
        self.remove_agent = remove_agent
        self.push_agent = push_agent

        # ---- 调试开关 ----
        self.verbose = bool(verbose)

        # 判定阈值（平面距离，做平方比较更省）
        self.clear_radius = float(clear_radius)
        self.r2 = self.clear_radius ** 2

        # 复检与驻留
        self.check_period = int(check_period)
        self.stop_sensing_after_push = bool(stop_sensing_after_push)
        self.min_dwell_steps = int(min_dwell_steps)

        self.button_name = button_name
        self.cup_names = list(cup_names or ["Cup1", "Cup_block"])
        self.button = None
        self.cups = []

        # 新增：把可调参挂到实例上，供 _resolve_handles / _sense_has_cup 使用
        self.z_tol = float(z_tol)
        self.priority_z_tol = float(priority_z_tol) if priority_z_tol is not None else max(self.z_tol, 0.12)
        self.center_clear_radius = float(center_clear_radius)
        self.r_btn = float(r_btn)
        self.priority_names = list(priority_names or [])

        # 运行态
        self.mode = "removal"          # 初始保守
        self._episode_step = 0
        self._last_check_step = -10**9
        self._sensing_disabled = False
        self._steps_in_mode = 0
        self.switch_count = 0
        self._has_cup_cached = 1
        self._mode_int = 0

        # ✅ 新增：首切步数 & 各agent调用计数
        self.first_switch_step = -1
        self._num_calls_removal = 0
        self._num_calls_push = 0

    def _resolve_handles(self):
        try:
            self.button = Shape(self.button_name)
            bx, by, bz = self.button.get_position()  # ← 先拿到坐标
            print(f"[Router] ✅ 按钮 '{self.button_name}' 位置=({bx:.3f}, {by:.3f}, {bz:.3f})")
        except Exception:
            self.button = None
            print(f"[Router] ⚠️ 按钮对象 '{self.button_name}' 未找到！")
            self.cups = []
            return
        
            # 2) Z 轴窗口（只考虑与按钮高度接近的物体）
      #  z_tol = getattr(self, "z_tol", 0.08)  # 允许通过 cfg 注入：router.z_tol
        z_min, z_max = bz - self.z_tol, bz + self.z_tol

        # 3) 收集候选：显式配置 + 自动枚举 Cup_copyX(_visible)
        #    - 显式配置有优先
        found = []
        names_seen = set()

        def _try_add(name: str):
            if name in names_seen:
                return
            try:
                h = Shape(name)
                _, _, z = h.get_position()
                if z_min <= z <= z_max:
                    found.append(h)
                    names_seen.add(name)
                    # 调试打印
                    # print(f"[Router][keep] {name}: z={z:.3f} in [{z_min:.3f},{z_max:.3f}]")
                    if self.verbose:
                        print(f"[Router][keep] {name}: z={z:.3f} in [{z_min:.3f},{z_max:.3f}]")
                else:
                    print(f"[Router][drop(z)] {name}: z={z:.3f} not in [{z_min:.3f},{z_max:.3f}]")
                    pass
            except Exception:
               pass

        # 3.1 显式配置的名字（如果你命令行传了 router.cup_names=...）
        for n in self.cup_names:
            _try_add(n)

        # 3.2 自动枚举：Cup_copyX 与 Cup_copyX_visible（X=0..9，可按需加大）
        for i in range(10):
            _try_add(f"Cup_copy{i}")
            _try_add(f"Cup_copy{i}_visible")

        # ✅ 自动枚举：Cup_block 变体
        for n in ("Cup_block", "Cup_block_visible", "CupBlock", "CupBlock_visible", "Cup_blockvisible", "Cupblockvisible"):
           _try_add(n)

        # 4) 名称再过滤：只保留匹配 ^Cup_copy\d+(_visible)?$ 的（以及显式给的）
        #    这样像 Cup1（地面的原始杯子）不会被留下
        cupcopy_pat = re.compile(r"^Cup_copy\d+(?:_visible)?$")
        # 过滤器也放宽：把 cupblock_pat 改为同时匹配两类写法
        cupblock_pat = re.compile(r"^Cup[_]?block(?:_?visible|_visible)?(?:#\d+)?$", re.IGNORECASE)
        
        final = []
        for h in found:
            try:
                name = h.get_name()
                if (name in self.cup_names) or cupcopy_pat.match(name) or cupblock_pat.match(name):
                    final.append(h)
            except Exception:
                continue


        self.cups = final
        print(f"[Router] ✅ 桌面候选杯子 {len(self.cups)} 个："
              + (", ".join([c.get_name() for c in self.cups]) if self.cups else "[无]"))
        


    def on_episode_reset(self):
        self.mode = "removal"
        self._episode_step = 0
        self._last_check_step = -10**9
        self._sensing_disabled = False
        self._steps_in_mode = 0
        self.switch_count = 0
        self._has_cup_cached = 1
        self._mode_int = 0
        self.first_switch_step = -1
        self._num_calls_removal = 0
        self._num_calls_push = 0
        self._resolve_handles()

    def _sense_has_cup(self) -> bool:
        if self.button is None:
            print("[Router] ⚠️ 未拿到按钮句柄，保持 removal，不进行切换。")
            return True
        bx, by, bz = self.button.get_position()
            # ---- 参数：高度容差、按钮有效半径、间隙阈值 ----
        z_tol = getattr(self, "z_tol", 0.08)   # 与 _resolve_handles 一致
        r_btn = getattr(self, "r_btn", 0.02)   # 按钮半径约 2cm，可按实际按钮直径调整
        # clear_radius 改作“边-边允许的最小安全间隙”，默认 0.03~0.04 更合理
        clearance_gap = min(self.clear_radius, 0.04)
        center_clear_r2 = self.center_clear_radius ** 2

        priority_names = set(self.priority_names)         # 例如 ["Cup_block", "Cup_blockvisible"]
        center_clear_radius = float(self.center_clear_radius)
        center_clear_r2 = center_clear_radius * center_clear_radius
        priority_z_tol = float(self.priority_z_tol) 

        # 小工具：估计杯子半径
        def est_cup_radius(shape_obj):
            try:
                bb = shape_obj.get_bounding_box()
                return 0.5 * max(bb[1]-bb[0], bb[3]-bb[2])
            except Exception:
                return 0.04


        # ✅ 1) 优先物体：不依赖 self.cups，直接按名字尝试拿句柄（含可见变体）
        for base in priority_names:
            for cand in (base, f"{base}_visible"):
                try:
                    h = Shape(cand)
                    x, y, z = h.get_position()
                    dz = abs(z - bz)
                    if dz > self.priority_z_tol:
                        if self.verbose:
                            print(f"[Router][PRIO-skip-z] {cand}: Δz={dz:.3f}m > {self.priority_z_tol:.3f}m")
                        continue
                    dx, dy = x - bx, y - by
                    d2 = dx*dx + dy*dy
                    if self.verbose:
                        print(f"[Router][PRIO-check] {cand}: center_dist={d2**0.5:.3f}m (th={self.center_clear_radius:.3f}), Δz={dz:.3f}")
                    if d2 < center_clear_r2:
                        print(f"[Router][PRIORITY HIT] {cand}: center_dist={d2**0.5:.3f}m < {self.center_clear_radius:.3f}m")
                        return True
                except Exception:
                    # 没有这个名字的对象就跳过
                    if self.verbose:
                        print(f"[Router][PRIO-miss] {cand}: not found")
                    continue


        for c in self.cups:
            try:
                name = c.get_name()
                x, y, z = c.get_position()
                dz = abs(z - bz)
                if dz > z_tol:
                    # 高度不在同一平面，忽略
                    continue

                # 中心-中心平面距离
                dx, dy = x - bx, y - by
                d_xy = (dx * dx + dy * dy) ** 0.5

                # 估计杯子半径（用 bbox 的 xy 最大半径）
                try:
                    bb = c.get_bounding_box()  # [xmin,xmax,ymin,ymax,zmin,zmax]
                    r_cup = 0.5 * max(bb[1] - bb[0], bb[3] - bb[2])
                except Exception:
                    r_cup = 0.04  # 兜底半径 4cm

                edge_gap = d_xy - r_cup - r_btn

                # 可开/关的调试输出（调参期建议打开）
                print(f"[Router][check] {name}: d_xy={d_xy:.3f}m, r_cup~{r_cup:.3f}m, "
                      f"r_btn={r_btn:.3f}m, edge_gap={edge_gap:.3f}m, Δz={dz:.3f}m")

                if edge_gap < clearance_gap:
                    print(f"[Router][HIT] {name}: edge_gap={edge_gap:.3f}m < {clearance_gap:.3f}m "
                          f"(d_xy={d_xy:.3f}, r_cup={r_cup:.3f}, r_btn={r_btn:.3f}, Δz={dz:.3f})")
                    return True
                
                # 在 for c in self.cups 循环的末尾，加一个兜底：
                if d_xy < center_clear_radius:
                    if self.verbose:
                        print(f"[Router][FALLBACK-HIT] {name}: center_dist={d_xy:.3f}m < {self.center_clear_radius:.3f}m")
                    return True

            except Exception:
                continue

        return False

    def _maybe_update_mode(self):
        need_check = False
        if self._episode_step == 0:
            need_check = True
        elif (self.mode == "removal") and (not self._sensing_disabled):
            if (self._episode_step - self._last_check_step) >= self.check_period:
                need_check = True
        if self._sensing_disabled:
            need_check = False

        if need_check:
            has_cup = self._sense_has_cup()
            self._has_cup_cached = int(has_cup)
            self._last_check_step = self._episode_step
            can_switch = (self._steps_in_mode >= self.min_dwell_steps)

            prev = self.mode
            if prev == "removal":
                if (not has_cup) and can_switch:
                    self.mode = "push"
                    self._steps_in_mode = 0
                    self.switch_count += 1
                    if self.first_switch_step < 0:
                        self.first_switch_step = self._episode_step  # ✅ 记录首次切换步
                    print(f"[Router] ✅ removal → push (step={self._episode_step})")
                    if self.stop_sensing_after_push:
                        self._sensing_disabled = True
            else:
                if has_cup and can_switch:
                    self.mode = "removal"
                    self._steps_in_mode = 0
                    self.switch_count += 1
                    print(f"[Router] 🔁 push → removal (step={self._episode_step})")

        self._mode_int = 0 if self.mode == "removal" else 1

    def act(self, obs):
        self._maybe_update_mode()

        if self.verbose:
            print(f"[Router] step={self._episode_step:04d} mode={self.mode} has_cup={self._has_cup_cached}")

        if self.mode == "removal":
            self._num_calls_removal += 1      # ✅ 计数
            action = self.remove_agent.act(obs)
        else:
            self._num_calls_push += 1         # ✅ 计数
            action = self.push_agent.act(obs)

        self._episode_step += 1
        self._steps_in_mode += 1
        return action


class RouterWorkspace(Workspace):
    def __init__(self, cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.snapshot = None
        super().__init__(cfg, train=False)

        # -------- 加载 checkpoint（绝对路径） --------
        rem_path = to_absolute_path(cfg.router.removal_snapshot)
        push_path = to_absolute_path(cfg.router.push_snapshot)

        rem_ckpt = torch.load(rem_path, map_location="cpu")
        push_ckpt = torch.load(push_path, map_location="cpu")

        # ✅ 打印两个 ckpt 指纹，便于确认“到底加载了谁”
        rem_fp = _state_dict_fingerprint(rem_ckpt["agent_state_dict"])
        push_fp = _state_dict_fingerprint(push_ckpt["agent_state_dict"])
        print(f"[Router] removal ckpt: {rem_path}  (fp={rem_fp})")
        print(f"[Router] push    ckpt: {push_path}  (fp={push_fp})")

        # -------- 实例化两个 agent --------
        rem_agent = hydra.utils.instantiate(cfg.method, accelerator=self.accelerator)
        rem_agent.load_state_dict(rem_ckpt["agent_state_dict"])
        rem_agent.eval()

        push_agent = hydra.utils.instantiate(cfg.method, accelerator=self.accelerator)
        push_agent.load_state_dict(push_ckpt["agent_state_dict"])
        push_agent.eval()

        self._episode_new_flag = True

        # -------- 初始化 Router（增加 verbose 开关） --------
        self.router = TwoAgentRouter(
            remove_agent=rem_agent,
            push_agent=push_agent,
            button_name=cfg.router.get("button_name", "push_button_target"),
            cup_names=cfg.router.get("cup_names", ["Cup1", "Cup_block"]),
            clear_radius=cfg.router.get("clear_radius", 0.12),
            check_period=cfg.router.get("check_period", 30),
            stop_sensing_after_push=cfg.router.get("stop_sensing_after_push", True),
            min_dwell_steps=cfg.router.get("min_dwell_steps", 0),
            verbose=cfg.router.get("verbose", False),  
            z_tol=cfg.router.get("z_tol", 0.08),
            priority_z_tol=cfg.router.get("priority_z_tol", None),
            center_clear_radius=cfg.router.get("center_clear_radius", 0.12),
            r_btn=cfg.router.get("r_btn", 0.02),
            priority_names=cfg.router.get("priority_names", []),
        )

    def _perform_env_steps(self, observations, info, env, eval_mode=True):
        metrics = {}

        if self._episode_new_flag:
            self.router.on_episode_reset()
            self._episode_new_flag = False

        with torch.no_grad():
            obs_torch = {k: torch.from_numpy(v).to(self.device) for k, v in observations.items()}
            if eval_mode:
                obs_torch = {k: v.unsqueeze(0) for k, v in obs_torch.items()}

            action = self.router.act(obs_torch)
            action = action.cpu().numpy()[0]

        # ✅ 增加更多指标，便于 post-hoc 判断是否真的在用 push
        metrics.update({
            "router/mode": int(self.router._mode_int),
            "router/has_cup": int(self.router._has_cup_cached),
            "router/switch_count": int(self.router.switch_count),
            "router/episode_step": int(self.router._episode_step),
            "router/first_switch_step": int(self.router.first_switch_step),
            "router/calls_removal": int(self.router._num_calls_removal),
            "router/calls_push": int(self.router._num_calls_push),
        })

        next_observation, reward, termination, truncation, next_info = env.step(action)

        if bool(termination) or bool(truncation):
            # ✅ 结束时打印一行总结（本 episode 是否用了 push）
            print(f"[Router][episode_end] switch_count={self.router.switch_count} "
                  f"first_switch_step={self.router.first_switch_step} "
                  f"calls: removal={self.router._num_calls_removal}, push={self.router._num_calls_push}")
            self._episode_new_flag = True

        return action, (next_observation, reward, termination, truncation, next_info), metrics
