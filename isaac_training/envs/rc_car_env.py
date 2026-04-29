from __future__ import annotations
from collections.abc import Sequence
import os
import torch

import isaaclab.sim as sim_utils # type: ignore
from isaaclab.actuators import ImplicitActuatorCfg # type: ignore
from isaaclab.assets import Articulation, ArticulationCfg # type: ignore
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg # type: ignore
from isaaclab.scene import InteractiveSceneCfg # type: ignore
from isaaclab.sim import SimulationCfg # type: ignore
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane # type: ignore
from isaaclab.utils import configclass   # type: ignore

_HERE = os.path.dirname(os.path.abspath(__file__))

@configclass
class RCCarEnvCfg(DirectMARLEnvCfg):
    """
    Config class to set up the environments including Scenes and Robots
    """

    # env
    decimation = 8 # how often the agent makes a decision, in this case 100Hz / 2
    episode_length_s = 30.0
    possible_agents = ["runner", "chaser"]
    action_spaces = {"runner": 2, "chaser": 2} # drive velocity + steer angle
    observation_spaces = {"runner": 4, "chaser": 4} # heading + target x and y + distance
    state_space = 0 # not relevant but needed

    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=decimation)

    _BASE_CAR_CFG = ArticulationCfg(
        prim_path="/World/envs/env_.*/PLACEHOLDER",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.07)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_HERE, "../rc_car.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        actuators={
            "rear_wheels": ImplicitActuatorCfg(
                joint_names_expr=["rear_left_wheel_joint", "rear_right_wheel_joint"],
                stiffness=0.0,
                damping=2.0,
            ),
            "front_steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_steer_joint", "front_right_steer_joint"],
                stiffness=100.0,
                damping=10.0,
            ),
            "front_wheels": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_joint", "front_right_wheel_joint"],
                stiffness=0.0,
                damping=0.1,
            ),
        },
    )
    runner: ArticulationCfg = _BASE_CAR_CFG.replace(
        prim_path="/World/envs/env_.*/Runner",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.07),
        ),
    )

    chaser: ArticulationCfg = _BASE_CAR_CFG.replace(
        prim_path="/World/envs/env_.*/Chaser",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.0, 0.0, 0.07),
        ),
    )
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,        # run 64 parallel environments during training
        env_spacing=12.0,   # space them 12 meters apart
        replicate_physics=True,
        clone_in_fabric=True,
    )
    
    # fixed car parameters
    drive_speed    = 37.5 # 1.5 m/s / 0.04m (wheel raduis)
    steering_angle = 0.5

    # environment Boundaries
    out_of_bounds_distance = 4.0
    car_spawn_range = 2.0

    # reward parameters
    reach_threshold = 0.5
    reach_bonus = 10.0
    out_of_bounds_penalty = 250.0
    step_penalty = 0.5

@configclass
class RCCarEvalEnvCfg(RCCarEnvCfg):
    """
    This config class allows for displaying multiple environments
    Currently we are only displaying one because of other issues so it's not very relevant
    """
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,       
        env_spacing=4.0,   
        replicate_physics=False,
        clone_in_fabric=False,
    )

class RCCarEnv(DirectMARLEnv):
    cfg: RCCarEnvCfg

    def __init__(self, cfg: RCCarEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Look up joind indices once at init time
        self._runner_rear_wheel_ids, _ = self.runner.find_joints(
            ["rear_left_wheel_joint", "rear_right_wheel_joint"]
        )
        self._runner_front_steer_ids, _ = self.runner.find_joints(
            ["front_left_steer_joint", "front_right_steer_joint"]
        )
        self._chaser_rear_wheel_ids, _ = self.chaser.find_joints(
            ["rear_left_wheel_joint", "rear_right_wheel_joint"]
        )
        self._chaser_front_steer_ids, _ = self.chaser.find_joints(
            ["front_left_steer_joint", "front_right_steer_joint"]
        )

        # Place to store the previous distance to use for the reward function
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        import omni.usd # type: ignore
        self.runner = Articulation(self.cfg.runner)
        self.chaser = Articulation(self.cfg.chaser)

        # Adds friction to ground
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            )
        )
        
        self.scene.clone_environments(copy_from_source=False) # clones the robot to all env while keeping them independent
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["runner"] = self.runner
        self.scene.articulations["chaser"] = self.chaser
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Spawn material at its own path
        wheel_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
        )
        wheel_material_cfg.func("/World/Physics/WheelMaterial", wheel_material_cfg)

        # Bind material to all wheel prims
        from pxr import UsdShade #type: ignore
        stage = omni.usd.get_context().get_stage()
        material = UsdShade.Material(stage.GetPrimAtPath("/World/Physics/WheelMaterial"))
        for i in range(self.num_envs):
            for wheel in ["rear_left_wheel", "rear_right_wheel", "front_left_wheel", "front_right_wheel"]:
                runner_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Runner/{wheel}")
                if runner_prim.IsValid():
                    UsdShade.MaterialBindingAPI(runner_prim).Bind(
                        material,
                        UsdShade.Tokens.strongerThanDescendants,
                        "physics"
                    )
                chaser_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Chaser/{wheel}")
                if chaser_prim.IsValid():
                    UsdShade.MaterialBindingAPI(chaser_prim).Bind(
                        material,
                        UsdShade.Tokens.strongerThanDescendants,
                        "physics"
                    )

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """
        Clamps forward/backward to [-1,1]
        Clamps steering angle to [-0.5, 0.5]
        This can be cleaner by extracting actions[runner] and actions[chaser] first
        """

        steer_angle = self.cfg.steering_angle
        actions['runner'][:, 0] = torch.clamp(actions['runner'][:, 0], -1, 1)
        actions['chaser'][:, 0] = torch.clamp(actions['chaser'][:, 0], -1, 1)
        actions['runner'][:, 1] = torch.clamp(actions['runner'][:, 1], -steer_angle, steer_angle)
        actions['chaser'][:, 1] = torch.clamp(actions['chaser'][:, 1], -steer_angle, steer_angle)

        self.runner_actions = actions['runner'].clone()
        self.chaser_actions = actions['chaser'].clone()

    def _apply_action(self):
        """
        Actions
        first column: forward/backward velocity
        second column: steering angle
        """

        runner_drive = self.runner_actions[:, 0] * self.cfg.drive_speed
        runner_steer = self.runner_actions[:, 1]

        chaser_drive = self.chaser_actions[:, 0] * self.cfg.drive_speed
        chaser_steer = self.chaser_actions[:, 1]

        # Both rear wheels get the same speed
        runner_rear_drive  = runner_drive.unsqueeze(1).repeat(1, 2)
        chaser_rear_drive  = chaser_drive.unsqueeze(1).repeat(1, 2)

        # Both front wheels get the same steering angle
        runner_front_steer = runner_steer.unsqueeze(1).repeat(1, 2)
        chaser_front_steer = chaser_steer.unsqueeze(1).repeat(1, 2)

        self.runner.set_joint_velocity_target(
            runner_rear_drive,
            joint_ids=self._runner_rear_wheel_ids
        )
        self.runner.set_joint_position_target(
            runner_front_steer,
            joint_ids=self._runner_front_steer_ids
        )

        self.chaser.set_joint_velocity_target(
            chaser_rear_drive,
            joint_ids=self._chaser_rear_wheel_ids
        )
        self.chaser.set_joint_position_target(
            chaser_front_steer,
            joint_ids=self._chaser_front_steer_ids
        )

    def _get_observations(self) -> dict:
        # Robot position and heading
        runner_pos = self.runner.data.root_pos_w[:, :2]         # shape (64, 2):  x, y
        chaser_pos = self.chaser.data.root_pos_w[:, :2]

        runner_quat = self.runner.data.root_quat_w
        runner_siny = 2.0 * (runner_quat[:, 0] * runner_quat[:, 3] + runner_quat[:, 1] * runner_quat[:, 2])
        runner_cosy = 1.0 - 2.0 * (runner_quat[:, 2] ** 2 + runner_quat[:, 3] ** 2)
        runner_heading = torch.atan2(runner_siny, runner_cosy).unsqueeze(1)  # shape (64, 1):  yaw

        # Vector to target
        runner_sin_yaw = torch.sin(runner_heading)
        runner_cos_yaw = torch.cos(runner_heading)

        runner_to_target = chaser_pos - runner_pos                 # shape (64, 2)
        runner_local_x =  runner_cos_yaw * runner_to_target[:, 0:1] + runner_sin_yaw * runner_to_target[:, 1:2]
        runner_local_y = -runner_sin_yaw * runner_to_target[:, 0:1] + runner_cos_yaw * runner_to_target[:, 1:2]
        runner_local_to_target = torch.cat([runner_local_x, runner_local_y], dim=1)

        runner_distance = torch.norm(runner_to_target, dim=1, keepdim=True)   # shape (64, 1)

        chaser_quat = self.chaser.data.root_quat_w
        chaser_siny = 2.0 * (chaser_quat[:, 0] * chaser_quat[:, 3] + chaser_quat[:, 1] * chaser_quat[:, 2])
        chaser_cosy = 1.0 - 2.0 * (chaser_quat[:, 2] ** 2 + chaser_quat[:, 3] ** 2)
        chaser_heading = torch.atan2(chaser_siny, chaser_cosy).unsqueeze(1)  # shape (64, 1):  yaw

        # Vector to target
        chaser_sin_yaw = torch.sin(chaser_heading)
        chaser_cos_yaw = torch.cos(chaser_heading)

        chaser_to_target = runner_pos - chaser_pos                 # shape (64, 2)
        chaser_local_x =  chaser_cos_yaw * chaser_to_target[:, 0:1] + chaser_sin_yaw * chaser_to_target[:, 1:2]
        chaser_local_y = -chaser_sin_yaw * chaser_to_target[:, 0:1] + chaser_cos_yaw * chaser_to_target[:, 1:2]
        chaser_local_to_target = torch.cat([chaser_local_x, chaser_local_y], dim=1)

        chaser_distance = torch.norm(chaser_to_target, dim=1, keepdim=True)   # shape (64, 1)

        runner_obs = torch.cat([runner_heading, runner_local_to_target, runner_distance], dim=1)
        chaser_obs = torch.cat([chaser_heading, chaser_local_to_target, chaser_distance], dim=1)

        return {'runner':runner_obs, 'chaser':chaser_obs}
    
    # Some library requires a public get_observations
    def get_observations(self) -> dict:
        return self._get_observations()

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        runner_pos = self.runner.data.root_pos_w[:, :2]
        chaser_pos = self.chaser.data.root_pos_w[:, :2]
        distance = torch.norm(runner_pos - chaser_pos, dim=1)

        distance_change = self.prev_distance - distance
        chaser_reward = torch.exp(-distance * 2.0) + distance_change * 2.0

        # Bonus for reaching the target
        reached = distance < self.cfg.reach_threshold
        chaser_reward[reached] += self.cfg.reach_bonus

        # Add per step penalty
        chaser_reward -= self.cfg.step_penalty

        # Currently the runner reward is just the opposite of the chaser
        runner_reward = -chaser_reward

        # Large penalty for going out of bounds
        env_origins_2d = self.scene.env_origins[:, :2]
        runner_dist_from_origin = torch.norm(runner_pos - env_origins_2d, dim=1)
        runner_out_of_bounds = runner_dist_from_origin > self.cfg.out_of_bounds_distance
        runner_out_of_bounds_penalty = runner_out_of_bounds.float() * self.cfg.out_of_bounds_penalty
        runner_reward -= runner_out_of_bounds_penalty

        chaser_dist_from_origin = torch.norm(chaser_pos - env_origins_2d, dim=1)
        chaser_out_of_bounds = chaser_dist_from_origin > self.cfg.out_of_bounds_distance
        chaser_out_of_bounds_penalty = chaser_out_of_bounds.float() * self.cfg.out_of_bounds_penalty
        chaser_reward -= chaser_out_of_bounds_penalty

        self.prev_distance = distance.clone()

        self.extras["log"] = {
            "chaser_reward": chaser_reward.mean(),
            "runner_reward": runner_reward.mean(),
        }

        return {'runner':runner_reward, 'chaser':chaser_reward}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns done when Robot reaches target or runs out of bounds
        """
        runner_pos = self.runner.data.root_pos_w[:, :2]
        chaser_pos = self.chaser.data.root_pos_w[:, :2]
        distance = torch.norm(runner_pos - chaser_pos, dim=1)

        reached = distance < self.cfg.reach_threshold

        env_origins_2d = self.scene.env_origins[:, :2]
        runner_dist_from_origin = torch.norm(runner_pos - env_origins_2d, dim=1)
        chaser_dist_from_origin = torch.norm(chaser_pos - env_origins_2d, dim=1)

        out_of_bounds = (runner_dist_from_origin > self.cfg.out_of_bounds_distance) | (chaser_dist_from_origin > self.cfg.out_of_bounds_distance)

        terminated = reached | out_of_bounds
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset function to be called every time an episode is complete
        """
        if env_ids is None:
            env_ids = self.runner._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state with correct env origins
        runner_default_root_state = self.runner.data.default_root_state[env_ids].clone()    # getting robots default pose
        runner_default_root_state[:, :3] += self.scene.env_origins[env_ids]        # adjusting pose for each environment
        chaser_default_root_state = self.chaser.data.default_root_state[env_ids].clone()
        chaser_default_root_state[:, :3] += self.scene.env_origins[env_ids]

        random_offset = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 2.0
        runner_default_root_state[:, 0:2] = self.scene.env_origins[env_ids, :2] + random_offset * self.cfg.car_spawn_range
        random_offset = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 2.0
        chaser_default_root_state[:, 0:2] = self.scene.env_origins[env_ids, :2] + random_offset * self.cfg.car_spawn_range


        self.runner.write_root_pose_to_sim(runner_default_root_state[:, :7], env_ids)
        self.runner.write_root_velocity_to_sim(runner_default_root_state[:, 7:], env_ids)
        self.chaser.write_root_pose_to_sim(chaser_default_root_state[:, :7], env_ids)
        self.chaser.write_root_velocity_to_sim(chaser_default_root_state[:, 7:], env_ids)

        runner_pos = runner_default_root_state[:, :2]
        chaser_pos = chaser_default_root_state[:, :2]
        self.prev_distance = self.prev_distance.clone()
        self.prev_distance[env_ids] = torch.norm(runner_pos - chaser_pos, dim=1)
