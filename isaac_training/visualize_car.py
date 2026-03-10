import argparse
from isaaclab.app import AppLauncher # type: ignore

parser = argparse.ArgumentParser(description="Visualize RC car model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import isaaclab.sim as sim_utils # type: ignore
from isaaclab.actuators import ImplicitActuatorCfg # type: ignore
from isaaclab.assets import Articulation, ArticulationCfg # type: ignore 
from isaaclab.sim import SimulationContext, SimulationCfg # type: ignore

_HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    sim = SimulationContext(SimulationCfg(dt=0.01))

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/light", light_cfg)

    # RC car
    robot_cfg = ArticulationCfg(
        prim_path="/World/rc_car",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_HERE, "rc_car.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
        ),
        actuators={
            "rear_wheels": ImplicitActuatorCfg(
                joint_names_expr=["rear_left_wheel_joint", "rear_right_wheel_joint"],
                stiffness=0.0,
                damping=0.05,
            ),
            "front_steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_steer_joint", "front_right_steer_joint"],
                stiffness=10.0,
                damping=0.1,
            ),
        },
    )

    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    print("Car spawned. Use the Isaac Sim viewport to inspect it.")
    print("Press Ctrl+C to exit.")

    # Just run the sim so you can look around
    while simulation_app.is_running():
        sim.step()
        robot.update(sim.get_physics_dt())

main()
simulation_app.close()