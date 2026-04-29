import argparse
from isaaclab.app import AppLauncher # type: ignore

parser = argparse.ArgumentParser(description="Evaluate trained tag policies")
parser.add_argument("--chaser_checkpoint", type=str, required=True, help="Path to chaser_actor.pt")
parser.add_argument("--runner_checkpoint", type=str, required=True, help="Path to runner_actor.pt")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper # type: ignore
from envs.rc_car_env import RCCarEnv, RCCarEvalEnvCfg
from envs.single_agent_wrapper import SingleAgentWrapper
from train_cfg import train_cfg_dict


def load_policy(env, agent_name, checkpoint_path):
    """Build an actor with the correct architecture then load saved weights."""
    wrapped = SingleAgentWrapper(env, active_agent=agent_name, frozen_policy=None)
    wrapped_env = RslRlVecEnvWrapper(wrapped)
    runner = OnPolicyRunner(wrapped_env, copy.deepcopy(train_cfg_dict), log_dir=None, device="cuda:0")
    state_dict = torch.load(checkpoint_path, map_location="cuda:0")
    runner.alg.actor.load_state_dict(state_dict)
    policy = runner.alg.actor
    policy.eval()
    return policy


def main():
    env_cfg = RCCarEvalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = RCCarEnv(cfg=env_cfg, render_mode="human")

    chaser_policy = load_policy(env, "chaser", args_cli.chaser_checkpoint)
    runner_policy = load_policy(env, "runner", args_cli.runner_checkpoint)

    obs, _ = env.reset()

    print("Running evaluation. Press Ctrl+C to stop.")
    with torch.inference_mode():
        while simulation_app.is_running():
            chaser_actions = chaser_policy({"policy": obs["chaser"]})
            runner_actions = runner_policy({"policy": obs["runner"]})
            obs, _, _, _, _ = env.step({"chaser": chaser_actions, "runner": runner_actions})

    env.close()


main()
simulation_app.close()
