import argparse
from isaaclab.app import AppLauncher #type: ignore

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg #type: ignore
import os
_HERE = os.path.dirname(os.path.abspath(__file__))


cfg = UrdfConverterCfg(
    asset_path=os.path.join(_HERE, "rc_car.urdf"),
    usd_dir=_HERE,
    usd_file_name="rc_car.usd",
    fix_base=False,
    merge_fixed_joints=False,
    self_collision=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="velocity",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.05,
        ),
    ),
)

converter = UrdfConverter(cfg)
print(f"USD saved to: {converter.usd_path}")

simulation_app.close()