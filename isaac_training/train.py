import argparse
from isaaclab.app import AppLauncher # type: ignore

parser = argparse.ArgumentParser(description="Train RC car tag policies")
parser.add_argument("--num_envs",       type=int, default=64,   help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Training iterations per phase")
parser.add_argument("--num_phases",     type=int, default=10,   help="Training phases per scenario")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import copy
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper # type: ignore
from rsl_rl.runners import OnPolicyRunner

from envs.rc_car_env import RCCarEnv, RCCarEnvCfg, RCCarEnvCfgLargeRunner
from train_cfg import train_cfg_dict
from envs.single_agent_wrapper import SingleAgentWrapper


def train_policy(env, active_agent: str, frozen_policy, previous_policy, log_dir):
    wrapped     = SingleAgentWrapper(env, active_agent=active_agent, frozen_policy=frozen_policy)
    wrapped_env = RslRlVecEnvWrapper(wrapped)
    runner      = OnPolicyRunner(wrapped_env, copy.deepcopy(train_cfg_dict), log_dir=log_dir, device="cuda:0")

    if previous_policy is not None:
        runner.alg.actor.load_state_dict(previous_policy.state_dict())

    runner.learn(num_learning_iterations=args_cli.max_iterations)
    return runner.alg.actor


def train_scenario(env, phases, log_dir, chaser_name, runner_name):
    """
    Alternately train the chaser and runner for `phases` phases.
    Returns (runner_policy, chaser_policy).
    Policy files are saved as <name>_actor.pt in log_dir after each phase.
    """
    runner_policy = None
    chaser_policy = None

    for phase in range(phases):
        if phase % 2 == 0:
            print(f"\n=== Phase {phase}: training {chaser_name} ===")
            chaser_policy = train_policy(
                env, "chaser",
                frozen_policy=runner_policy,
                previous_policy=chaser_policy,
                log_dir=log_dir,
            )
            torch.save(chaser_policy.state_dict(), os.path.join(log_dir, f"{chaser_name}_actor.pt"))
        else:
            print(f"\n=== Phase {phase}: training {runner_name} ===")
            runner_policy = train_policy(
                env, "runner",
                frozen_policy=chaser_policy,
                previous_policy=runner_policy,
                log_dir=log_dir,
            )
            torch.save(runner_policy.state_dict(), os.path.join(log_dir, f"{runner_name}_actor.pt"))

    return runner_policy, chaser_policy


def reset_stage():
    """Drop all prims so the next env can build a fresh scene."""
    import omni.usd # type: ignore
    omni.usd.get_context().new_stage()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = os.path.join(os.path.dirname(__file__), "models/trained", f"rc_car_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Scenario A: small car is runner, large car is chaser
    # Trains: small_runner_policy and large_chaser_policy
    # ------------------------------------------------------------------
    print("\n========== Scenario A: small runner vs large chaser ==========")
    env_cfg_a = RCCarEnvCfg()
    env_cfg_a.scene.num_envs = args_cli.num_envs
    env_a = RCCarEnv(cfg=env_cfg_a)

    with torch.inference_mode(mode=False):
        train_scenario(
            env_a,
            phases=args_cli.num_phases,
            log_dir=log_dir,
            chaser_name="large_chaser",
            runner_name="small_runner",
        )

    env_a.close()
    reset_stage()

    # ------------------------------------------------------------------
    # Scenario B: large car is runner, small car is chaser
    # Trains: large_runner_policy and small_chaser_policy
    # ------------------------------------------------------------------
    print("\n========== Scenario B: large runner vs small chaser ==========")
    env_cfg_b = RCCarEnvCfgLargeRunner()
    env_cfg_b.scene.num_envs = args_cli.num_envs
    env_b = RCCarEnv(cfg=env_cfg_b)

    with torch.inference_mode(mode=False):
        train_scenario(
            env_b,
            phases=args_cli.num_phases,
            log_dir=log_dir,
            chaser_name="small_chaser",
            runner_name="large_runner",
        )

    env_b.close()

    print(f"\nAll policies saved to: {log_dir}")
    print("  small_runner_actor.pt  — small car playing runner")
    print("  large_chaser_actor.pt  — large car playing chaser")
    print("  large_runner_actor.pt  — large car playing runner")
    print("  small_chaser_actor.pt  — small car playing chaser")


main()
simulation_app.close()
