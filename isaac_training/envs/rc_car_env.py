from __future__ import annotations
from collections.abc import Sequence
import os
import torch

import isaaclab.sim as sim_utils # type: ignore
from isaaclab.actuators import ImplicitActuatorCfg # type: ignore
from isaaclab.assets import Articulation, ArticulationCfg # type: ignore
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg # type: ignore
from isaaclab.scene import InteractiveSceneCfg # type: ignore
from isaaclab.sim import SimulationCfg # type: ignore
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane # type: ignore
from isaaclab.utils import configclass   # type: ignore

_HERE = os.path.dirname(os.path.abspath(__file__))

@configclass
class RCCarEnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,       # run 64 parallel environments during training
        env_spacing=4.0,   # space them 4 meters apart
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_HERE, "../rc_car.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
    ),
    actuators={
        "rear_wheels": ImplicitActuatorCfg(
            joint_names_expr=["rear_left_wheel_joint", "rear_right_wheel_joint"],
            stiffness=0.0,
            damping=2.0,
        ),
        "front_steering": ImplicitActuatorCfg(
            joint_names_expr=["front_left_steer_joint", "front_right_steer_joint"],
            stiffness=10.0,
            damping=0.1,
        ),
    },
)
    
     # Fixed car parameters
    drive_speed    = 37.5 # 1.5 m/s / 0.04m (wheel raduis)
    steering_angle = 0.5

    # Environment
    decimation        = 2           # how often the agent makes a decision, in this case 100Hz / 2
    episode_length_s  = 30.0
    action_space      = 6           # see apply_action
    observation_space = 6           # x, y, heading + target x, y, distance
    state_space       = 0           # not relevant but needed

class RCCarEnv(DirectRLEnv):
    cfg: RCCarEnvCfg

    def __init__(self, cfg: RCCarEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Look up joind indices once at init time
        self._rear_wheel_ids, _ = self.robot.find_joints(
            ["rear_left_wheel_joint", "rear_right_wheel_joint"]
        )
        self._front_steer_ids, _ = self.robot.find_joints(
            ["front_left_steer_joint", "front_right_steer_joint"]
        )

        # Target position for each environment
        self.target_pos = torch.zeros(self.num_envs, 2, device=self.device) # we use 2 because we are finding (x,y) for each env
        self._randomize_targets()

    def _randomize_targets(self, env_ids=None):
        """Place targets at random positions within 1 meter."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Random x, y between -1 and 1
        self.target_pos[env_ids] = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 2.0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        self.scene.clone_environments(copy_from_source=False) # clones the robot to all env while keeping them independent
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        

    def _pre_physics_step(self, actions: torch.Tensor): # honestly I have no idea what this does
        """Convert discrete actions to wheel velocities."""
        self.actions = actions.clone()

    def _apply_action(self):
        """
        Actions:
        0 = forward
        1 = forward + left
        2 = forward + right
        3 = backward
        4 = backward + left
        5 = backward + right
        """
        speed = self.cfg.drive_speed
        angle = self.cfg.steering_angle

        action_idx = self.actions.argmax(dim=1)

        # Rear wheel velocity: positive = forward, negative = backward
        drive = torch.zeros(self.num_envs, device=self.device)
        drive[action_idx == 0] =  speed
        drive[action_idx == 1] =  speed
        drive[action_idx == 2] =  speed
        drive[action_idx == 3] = -speed
        drive[action_idx == 4] = -speed
        drive[action_idx == 5] = -speed

        # Front wheel steering angle: positive = left, negative = right
        steer = torch.zeros(self.num_envs, device=self.device)
        steer[action_idx == 1] =  angle
        steer[action_idx == 2] = -angle
        steer[action_idx == 4] =  angle
        steer[action_idx == 5] = -angle

        # Both rear wheels get the same speed
        rear_drive = drive.unsqueeze(1).repeat(1, 2)
        # Both front wheels get the same steering angle
        front_steer = steer.unsqueeze(1).repeat(1, 2)

        self.robot.set_joint_velocity_target(
            rear_drive,
            joint_ids=self._rear_wheel_ids
        )
        self.robot.set_joint_position_target(
            front_steer,
            joint_ids=self._front_steer_ids
        )

    def _get_observations(self) -> dict:
        # Robot position and heading
        robot_pos = self.robot.data.root_pos_w[:, :2]           # shape (64, 2) - x, y
        quat = self.robot.data.root_quat_w
        siny = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        robot_heading = torch.atan2(siny, cosy).unsqueeze(1)  # shape (64, 1) yaw

        # Vector to target
        sin_yaw = torch.sin(robot_heading)
        cos_yaw = torch.cos(robot_heading)

        to_target = self.target_pos - robot_pos                 # shape (64, 2)
        local_x =  cos_yaw * to_target[:, 0:1] + sin_yaw * to_target[:, 1:2]
        local_y = -sin_yaw * to_target[:, 0:1] + cos_yaw * to_target[:, 1:2]
        local_to_target = torch.cat([local_x, local_y], dim=1)

        distance = torch.norm(to_target, dim=1, keepdim=True)   # shape (64, 1)

        obs = torch.cat([robot_pos, robot_heading, local_to_target, distance], dim=1)
        return {"policy": obs}
    
    # Some library requires a public get_observations idk
    def get_observations(self) -> dict:
        return self._get_observations()

    def _get_rewards(self) -> torch.Tensor:
        robot_pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(self.target_pos - robot_pos, dim=1)

        # Reward for being close to the target
        reward = 1.0 / (1.0 + distance)

        # Bonus for reaching the target
        reached = distance < 0.3
        reward[reached] += 10.0

        # Small penalty each step to encourage speed
        reward -= 0.01

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(self.target_pos - robot_pos, dim=1)

        # Done if reached target or fell out of bounds
        reached = distance < 0.3
        out_of_bounds = torch.norm(robot_pos, dim=1) > 10.0

        terminated = reached | out_of_bounds
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state with correct env origins
        default_root_state = self.robot.data.default_root_state[env_ids]    # getting robots default pose
        default_root_state[:, :3] += self.scene.env_origins[env_ids]        # adjusting pose for each environment

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._randomize_targets(env_ids)