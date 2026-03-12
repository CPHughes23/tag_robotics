from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from envs.rc_car_env import RCCarEnv, RCCarEnvCfg

def main():
    env_cfg = RCCarEnvCfg()
    env_cfg.scene.num_envs = 1
    env = RCCarEnv(cfg=env_cfg)

    # Print initial state
    obs = env._get_observations()["policy"]
    print("\n--- INITIAL OBS ---")
    print(f"  robot pos:     {obs[0, 0:2]}")
    print(f"  heading:       {obs[0, 2]:.3f} rad")
    print(f"  local target:  {obs[0, 3:5]}")
    print(f"  distance:      {obs[0, 5]:.3f} m")
    print(f"  target_pos:    {env.target_pos[0]}")
    print(f"  env_origin:    {env.scene.env_origins[0, :2]}")
    print(f"  init_height:   {env.cfg.robot.init_state.pos}")

    # Drive forward for 100 steps and print position each time
    print("\n--- DRIVING FORWARD 100 STEPS ---")
    action = torch.zeros(1, 6)
    action[0, 0] = 1.0  # action 0 = forward

    for i in range(100):
        # forward action as one-hot, shape (1, 6)
        action = torch.zeros(1, 6, device=env.device)
        action[0, 0] = 1.0  # action 0 = forward

        obs, reward, terminated, truncated, info = env.step(action)

        if i % 20 == 0:
            pos = env.robot.data.root_pos_w[0, :2]
            dist = torch.norm(env.target_pos[0] - pos).item()
            print(f"  step {i:3d} | pos: {pos} | dist to target: {dist:.3f}")

    print("\n--- FINAL OBS ---")
    obs = env._get_observations()["policy"]
    print(f"  robot pos:     {obs[0, 0:2]}")
    print(f"  local target:  {obs[0, 3:5]}")
    print(f"  distance:      {obs[0, 5]:.3f} m")

    env.close()

main()
simulation_app.close()