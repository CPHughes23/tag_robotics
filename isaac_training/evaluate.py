import argparse
from isaaclab.app import AppLauncher # type: ignore

parser = argparse.ArgumentParser(description="Evaluate trained RC car policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to visualize")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper # type: ignore
from envs.rc_car_env import RCCarEnv, RCCarEvalEnvCfg
from train_cfg import train_cfg_dict

def main():
    try:
        env_cfg = RCCarEvalEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = RCCarEnv(cfg=env_cfg, render_mode="human")
        wrapped_env = RslRlVecEnvWrapper(env)

        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if "Robot" in prim.GetPath().pathString:
                print(prim.GetPath())

        # Load checkpoint
        checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
        print(f"Loaded checkpoint from iteration: {checkpoint['iter']}")

        # Reconstruct the actor from saved state
        runner = OnPolicyRunner(wrapped_env, train_cfg_dict, log_dir=None, device="cuda:0")
        runner.load(args_cli.checkpoint)
        policy = runner.alg.actor
        policy.eval()

        print("Running evaluation. Watch the Isaac Sim viewport.")
        print("Press Ctrl+C to stop.")

        with torch.inference_mode():
            while simulation_app.is_running():
                obs = wrapped_env.get_observations().to("cuda:0")
                # print(obs["policy"][:4])
                actions = runner.alg.act(obs)
                wrapped_env.step(actions)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

main()
simulation_app.close()