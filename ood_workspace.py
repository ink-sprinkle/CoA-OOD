def _spawn_random_obstacles(self):
    """
    Called after reset_to_demo(): randomly generate static obstacles.
    Returns (handles, meta) for tracking and removal.
    """
    enable = bool(self.cfg.get('obstacle_test', False))
    if not enable:
        return [], {"num": 0}

    n_min = int(self.cfg.get('obstacle_num_min', 1))
    n_max = int(self.cfg.get('obstacle_num_max', 3))
    table_z = float(self.cfg.get('obstacle_table_z', 0.75))
    size_cfg = self.cfg.get('obstacle_size', [0.05, 0.05, 0.10])
    size = OmegaConf.to_container(size_cfg, resolve=True)

    x_min, x_max = 0.18, 0.55
    y_min, y_max = -0.25, 0.25
    n_obs = int(np.random.randint(n_min, n_max + 1))
    handles = []

    for _ in range(n_obs):
        obs = Shape.create(
            type=PrimitiveShape.CUBOID,
            size=size,
            static=True,
            respondable=True
        )
        try:
            obs.set_color([1.0, 0.2, 0.2])
        except Exception:
            pass

        x = float(np.random.uniform(x_min, x_max))
        y = float(np.random.uniform(y_min, y_max))
        z = table_z
        obs.set_position([x, y, z])
        handles.append(obs)

        # Disable physics interaction (visual interference only)
        #try:
           # obs.set_dynamic(False)
           # obs.set_respondable(False)
           #obs.set_collidable(False)
       # except Exception:
          #  pass

    return handles, {"num": n_obs}


def _remove_obstacles(self, handles):
    """Remove obstacles after each episode."""
    if not handles:
        return
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def eval(self) -> Dict[str, Any]:
    # ...
    for episode_idx in range(num_episodes):
        try:
            # add obstacle
            obstacle_handles, obstacle_meta = [], {"num": 0}
            try:
                obstacle_handles, obstacle_meta = self._spawn_random_obstacles()
            except Exception as e:
                print("[obstacle] spawn failed:", e)

            # original eval logic↓
            # ...
            # (do evaluation steps, success counting, etc.)

        finally:
            # obstacle clear
            try:
                self._remove_obstacles(obstacle_handles)
            except Exception as e:
                print("[obstacle] remove failed:", e)



