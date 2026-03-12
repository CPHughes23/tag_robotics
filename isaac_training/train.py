import argparse
from isaaclab.app import AppLauncher # type: ignore

# This must be ran before other imports because they require the app running
parser = argparse.ArgumentParser(description="Train RC car to navigate to target")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
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




def main():
    # Set up environment config
    env_cfg = RCCarEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment
    env = RCCarEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)

    # Set up PPO training config
    train_cfg_dict = {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations,
        "save_interval": 100,
        "experiment_name": "rc_car",
        "empirical_normalization": False,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 64, 32],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "CategoricalDistribution",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 64, 32],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }

    # Create output directory for saved models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(__file__),
        "models/trained",
        f"rc_car_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create the PPO runner and start training
    runner = OnPolicyRunner(wrapped_env, train_cfg_dict, log_dir=log_dir, device="cuda:0")

    print(f"\nStarting training with {args_cli.num_envs} parallel environments")
    print(f"Models will be saved to: {log_dir}")
    print(f"To monitor training, run: tensorboard --logdir {log_dir}\n")

    runner.learn(num_learning_iterations=args_cli.max_iterations)

    env.close()

main()
simulation_app.close()