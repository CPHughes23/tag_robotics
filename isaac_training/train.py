import argparse
from isaaclab.app import AppLauncher # type: ignore

# This must be ran before other imports because they require the app running
parser = argparse.ArgumentParser(description="Train RC car to navigate to target")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Path to the .pt checkpoint file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # run without GUI during training
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlVecEnvWrapper # type: ignore
from rsl_rl.runners import OnPolicyRunner
from envs.rc_car_env import RCCarEnv, RCCarEnvCfg
from train_cfg import train_cfg_dict
from isaaclab.envs import DirectMARLEnv
from envs.single_agent_wrapper import SingleAgentWrapper
import copy


def train_policy(env, active_agent: str, frozen_policy, previous_policy, log_dir):
    # print("----------------- Env Printing from train_policy--------------------")
    # print(type(env))
    # print(hasattr(env, 'scene'))

    wrapped = SingleAgentWrapper(env, active_agent=active_agent, frozen_policy=frozen_policy)
    wrapped_env = RslRlVecEnvWrapper(wrapped)

    # print(train_cfg_dict["algorithm"])

    runner = OnPolicyRunner(wrapped_env, copy.deepcopy(train_cfg_dict), log_dir=log_dir, device="cuda:0")

    if previous_policy is not None:
        runner.alg.actor.load_state_dict(previous_policy.state_dict())

    runner.learn(num_learning_iterations=args_cli.max_iterations)

    new_policy = runner.alg.actor

    return new_policy

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__),
        "models/trained",
        f"rc_car_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    chaser_policy = None
    runner_policy = None

    env_cfg = RCCarEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = RCCarEnv(cfg=env_cfg)

    for phase in range(10):
        with torch.inference_mode(mode=False):
            if phase%2 == 0:
                chaser_policy = train_policy(env, "chaser", frozen_policy=runner_policy, previous_policy=chaser_policy, log_dir=log_dir)
                torch.save(chaser_policy.state_dict(), os.path.join(log_dir, "chaser_actor.pt"))
            else:
                runner_policy = train_policy(env, "runner", frozen_policy=chaser_policy, previous_policy=runner_policy, log_dir=log_dir)
                torch.save(runner_policy.state_dict(), os.path.join(log_dir, "runner_actor.pt"))

    env.close()

main()
simulation_app.close()