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
    """
    Config class to set up the environments including Scenes and Robots
    """
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,        # run 64 parallel environments during training
        env_spacing=12.0,   # space them 12 meters apart
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.07),
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
    
    # Fixed car parameters
    drive_speed    = 37.5 # 1.5 m/s / 0.04m (wheel raduis)
    steering_angle = 0.5

    # Environment
    decimation        = 8           # how often the agent makes a decision, in this case 100Hz / 2
    episode_length_s  = 30.0
    action_space      = 2           # drive velocity + steer angle
    observation_space = 4           # heading + target x and y + distance
    state_space       = 0           # not relevant but needed

    # Environment Boundaries
    out_of_bounds_distance = 4.0
    target_spawn_range = 2.0

    # Reward parameters
    reach_threshold = 0.5
    reach_bonus = 10.0
    out_of_bounds_penalty = 250.0
    step_penalty = 0.5

@configclass
class RCCarEvalEnvCfg(RCCarEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,       # run 64 parallel environments during training
        env_spacing=4.0,   # space them 4 meters apart
        replicate_physics=False,
        clone_in_fabric=False,
    )

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
        self.target_pos = torch.zeros(self.num_envs, 2, device=self.device) # We use 2 because we are finding (x,y) for each env
        self._randomize_targets()
        self._markers_initialized = False # This ensures we only create the markers once

        # Place to store the previous distance to use for the reward function
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

    def _randomize_targets(self, env_ids=None):
        """Place targets at random positions within 4 meters."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        random_offset = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 2.0
        self.target_pos[env_ids] = self.scene.env_origins[env_ids, :2] + random_offset * self.cfg.target_spawn_range

    def _setup_scene(self):
        import omni.usd # type: ignore
        self.robot = Articulation(self.cfg.robot)

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
        self.scene.articulations["robot"] = self.robot
        
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
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/{wheel}")
                if prim.IsValid():
                    UsdShade.MaterialBindingAPI(prim).Bind(
                        material,
                        UsdShade.Tokens.strongerThanDescendants,
                        "physics"
                    )

    def _pre_physics_step(self, actions: torch.Tensor):
        if not hasattr(self, '_action_map'):
            self._action_map = torch.tensor([
                [ 1.0,  0.0], [ 1.0, -0.5], [ 1.0,  0.5],
                [-1.0,  0.0], [-1.0, -0.5], [-1.0,  0.5],
                [ 0.0,  0.0],
            ], device=self.device)

        # Treat action[:,0] as drive signal, action[:,1] as steer signal
        # Snap each to nearest discrete value
        drive = actions[:, 0]
        steer = actions[:, 1]

        drive_disc = torch.where(drive > 0.33, 1.0, torch.where(drive < -0.33, -1.0, 0.0))
        steer_disc = torch.where(steer > 0.17, 0.5, torch.where(steer < -0.17, -0.5, 0.0))

        self.actions = torch.stack([drive_disc, steer_disc], dim=1)

        # print("Discretized actions:", self.actions[:5])

    def _apply_action(self):
        """
        Actions
        first column: forward/backward velocity
        second column: steering angle
        """

        drive = self.actions[:, 0] * self.cfg.drive_speed
        steer = self.actions[:, 1]

        # Both rear wheels get the same speed
        rear_drive  = drive.unsqueeze(1).repeat(1, 2)
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

    def _update_target_markers(self):
        import omni.usd # type: ignore
        from pxr import Gf  # type: ignore
        stage = omni.usd.get_context().get_stage()

        if not self._markers_initialized:
            marker_cfg = sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
            for i in range(self.num_envs):
                path = f"/World/envs/env_{i}/Target"
                if not stage.GetPrimAtPath(path).IsValid():
                    marker_cfg.func(path, marker_cfg)
            self._markers_initialized = True

        # Update positions every call
        for i in range(self.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Target")
            if prim.IsValid():
                xform = prim.GetAttribute("xformOp:translate")
                if xform:
                    xform.Set(Gf.Vec3d(
                        float(self.target_pos[i, 0]),
                        float(self.target_pos[i, 1]),
                        0.1
                    ))

    def _get_observations(self) -> dict:
        # Robot position and heading
        robot_pos = self.robot.data.root_pos_w[:, :2]         # shape (64, 2):  x, y
        quat = self.robot.data.root_quat_w
        siny = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        robot_heading = torch.atan2(siny, cosy).unsqueeze(1)  # shape (64, 1):  yaw

        # Vector to target
        sin_yaw = torch.sin(robot_heading)
        cos_yaw = torch.cos(robot_heading)

        to_target = self.target_pos - robot_pos                 # shape (64, 2)
        local_x =  cos_yaw * to_target[:, 0:1] + sin_yaw * to_target[:, 1:2]
        local_y = -sin_yaw * to_target[:, 0:1] + cos_yaw * to_target[:, 1:2]
        local_to_target = torch.cat([local_x, local_y], dim=1)

        car_length = 0.32
        distance = torch.norm(to_target, dim=1, keepdim=True) / car_length  # shape (64, 1), normalized by car length
        print(f"Distance: {distance[:5]} car lengths")

        obs = torch.cat([robot_heading, local_to_target, distance], dim=1)
        
        if self.render_mode is not None:
            self._update_target_markers()

        return {"policy": obs}
    
    # Some library requires a public get_observations
    def get_observations(self) -> dict:
        return self._get_observations()

    def _get_rewards(self) -> torch.Tensor:
        robot_pos = self.robot.data.root_pos_w[:, :2]
        distance  = torch.norm(self.target_pos - robot_pos, dim=1)

        vel         = self.robot.data.root_lin_vel_w[:, :2]
        speed       = torch.norm(vel, dim=1)

        # --- Lazy init guard ---
        if not hasattr(self, 'prev_distance'):
            self.prev_distance  = distance.clone()
        if not hasattr(self, 'prev_speed'):
            self.prev_speed     = speed.clone()
        if not hasattr(self, 'prev_direction'):
            self.prev_direction = torch.zeros(self.num_envs, device=self.device)
        ang_vel     = self.robot.data.root_ang_vel_w[:, 2]

        quat    = self.robot.data.root_quat_w
        siny    = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy    = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        heading = torch.atan2(siny, cosy)

        heading_vec  = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
        signed_speed = (vel * heading_vec).sum(dim=1)   # + forward, - reverse

        #  1. potential shaping: smooth pull toward target, no sign flips
        potential      = torch.exp(-distance * 1.5)
        prev_potential = torch.exp(-self.prev_distance * 1.5)
        reward = (potential - prev_potential) * 4.0

        #  2. heading reward: only fires when actually moving
        to_target    = self.target_pos - robot_pos
        target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
        angle_error  = torch.atan2(
            torch.sin(target_angle - heading),
            torch.cos(target_angle - heading)
        )
        forward_align = torch.cos(angle_error)
        reverse_align = torch.cos(angle_error + torch.pi)
        best_align    = torch.max(forward_align, reverse_align)

        # Only reward heading when moving AND far enough to matter
        # prevents obsessing over alignment at the goal
        moving          = speed > 0.05
        far_from_target = distance > self.cfg.reach_threshold * 2.0
        heading_scale   = torch.clamp(distance / 0.8, 0.0, 1.0)
        reward += best_align * heading_scale * moving.float() * far_from_target.float() * 0.4

        #  3. velocity toward target: reward purposeful movement
        # Project velocity directly onto the to-target vector (unit)
        to_target_dist = distance.unsqueeze(1).clamp(min=1e-6)
        to_target_unit = to_target / to_target_dist
        vel_toward      = (vel * to_target_unit).sum(dim=1)   # + = moving toward

        # Only reward this when not already at target
        reward += vel_toward * far_from_target.float() * 0.3

        #  4. goal zone: reward staying, punish leaving
        reached = distance < self.cfg.reach_threshold

        # Per-step bonus for being at goal (rewards fast arrival too,
        # since more steps at goal = more bonus)
        reward += reached.float() * self.cfg.reach_bonus

        # Punish any movement once inside the goal — kills the jitter loop
        reward -= reached.float() * speed * 2.0

        #  5. reversal penalty: break the forward/reverse oscillation
        direction = torch.sign(signed_speed)
        # Only count a reversal when actually moving both steps
        both_moving = (speed > 0.05) & (self.prev_speed > 0.05)
        reversal    = both_moving & ((direction * self.prev_direction) < 0)
        reward     -= reversal.float() * 0.3

        #  6. efficiency: small time penalty to encourage fast arrival
        # Scaled by distance so the penalty is lighter near the goal
        # (avoids punishing careful final approach)
        reward -= self.cfg.step_penalty * torch.clamp(distance / 2.0, 0.1, 1.0)

        #  7. out of bounds
        env_origins_2d   = self.scene.env_origins[:, :2]
        dist_from_origin = torch.norm(robot_pos - env_origins_2d, dim=1)
        out_of_bounds    = dist_from_origin > self.cfg.out_of_bounds_distance
        reward          -= out_of_bounds.float() * self.cfg.out_of_bounds_penalty

        #  8. for future calculations
        self.prev_distance  = distance.clone()
        self.prev_direction = direction.clone()
        self.prev_speed     = speed.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns done when Robot reaches target or runs out of bounds
        """
        robot_pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(self.target_pos - robot_pos, dim=1)

        reached = distance < self.cfg.reach_threshold

        env_origins_2d = self.scene.env_origins[:, :2]
        dist_from_origin = torch.norm(robot_pos - env_origins_2d, dim=1)
        out_of_bounds = dist_from_origin > self.cfg.out_of_bounds_distance

        terminated = reached | out_of_bounds
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset function to be called every time an episode is complete
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to default state with correct env origins
        default_root_state = self.robot.data.default_root_state[env_ids].clone()    # getting robots default pose
        default_root_state[:, :3] += self.scene.env_origins[env_ids]        # adjusting pose for each environment

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._randomize_targets(env_ids)

        robot_pos = default_root_state[:, :2]
        self.prev_distance[env_ids] = torch.norm(self.target_pos[env_ids] - robot_pos, dim=1)
