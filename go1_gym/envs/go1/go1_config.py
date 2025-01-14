from typing import Union

from params_proto import Meta

from go1_gym.envs.base.legged_robot_config import Cfg


def config_go1(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        "FL_hip_joint": 0.1,  # [rad]
        "RL_hip_joint": 0.1,  # [rad]
        "FR_hip_joint": -0.1,  # [rad]
        "RR_hip_joint": -0.1,  # [rad]
        "FL_thigh_joint": 0.8,  # [rad]
        "RL_thigh_joint": 1.0,  # [rad]
        "FR_thigh_joint": 0.8,  # [rad]
        "RR_thigh_joint": 1.0,  # [rad]
        "FL_calf_joint": -1.5,  # [rad]
        "RL_calf_joint": -1.5,  # [rad]
        "FR_calf_joint": -1.5,  # [rad]
        "RR_calf_joint": -1.5,  # [rad]
    }

    _ = Cnfg.control
    _.control_type = "P"
    _.stiffness = {"joint": 30.0}  # [N*m/rad]
    _.damping = {"joint": 1.0}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    # _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/a1_description/urdf/a1_with_head.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    # _.flip_visual_attachments = False
    _.flip_visual_attachments = True
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.34
    _.use_terminal_foot_height = False
    _.use_terminal_body_height = True
    _.terminal_body_height = 0.05
    _.use_terminal_roll_pitch = True
    _.terminal_body_ori = 1.6
    _.reward_container_name = "CoRLRewards"
    _.only_positive_rewards = False
    _.only_positive_rewards_ji22_style = True
    _.sigma_rew_neg = 0.02
    _.kappa_gait_probs = 0.07
    _.gait_force_sigma = 100.
    _.gait_vel_sigma = 10.

    _ = Cnfg.reward_scales
    _.torques = -0.0001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -5.0
    _.base_height = -30.0
    _.feet_slip = -0.04
    _.action_smoothness_1 = -0.1
    _.action_smoothness_2 = -0.1
    _.dof_vel = -1e-4
    _.dof_pos = -0.0
    _.jump = 10.0
    _.base_height = 0.0
    _.eight_target = 0.30
    _.estimation_bonus = 0.0
    _.raibert_heuristic = -10.0
    _.feet_impact_vel = -0.0
    _.feet_clearance = -0.0
    _.feet_clearance_cmd = -0.0
    _.feet_clearance_cmd_linear = -30.0
    _.feet_contact_forces = 0.0
    _.orientation = 0.0
    _.orientation_control = -5.0
    _.tracking_stance_width = -0.0
    _.tracking_stance_length = -0.0
    _.lin_vel_z = -0.02
    _.ang_vel_xy = -0.001
    _.feet_air_time = 0.0
    _.hop_symmetry = 0.0
    _.tracking_contacts_shaped_force = 4.0
    _.tracking_contacts_shaped_vel = 4.0
    _.collision = -5.0

    _ = Cnfg.terrain
    _.mesh_type = "trimesh"
    _.measure_heights = True
    _.terrain_noise_magnitude = 0.0
    _.border_size = 25
    _.x_init_range = 0.2
    _.y_init_range = 0.2
    _.teleport_thresh = 0.3
    _.teleport_robots = False
    _.center_span = 4
    _.horizontal_scale = 0.10
    _.yaw_init_range = 3.14
    # _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2, 0, 0, 0, 0.0]
    _.curriculum = True

    _ = Cnfg.env
    _.observe_vel = False
    _.num_envs = 4000
    _.priv_observe_motion = False
    _.priv_observe_gravity_transformed_motion = False
    _.priv_observe_friction_indep = False
    _.priv_observe_friction = True
    _.priv_observe_restitution = True
    _.priv_observe_base_mass = False
    _.priv_observe_gravity = False
    _.priv_observe_com_displacement = False
    _.priv_observe_ground_friction = False
    _.priv_observe_ground_friction_per_foot = False
    _.priv_observe_motor_strength = False
    _.priv_observe_motor_offset = False
    _.priv_observe_Kp_factor = False
    _.priv_observe_Kd_factor = False
    _.priv_observe_body_velocity = False
    _.priv_observe_body_height = False
    _.priv_observe_desired_contact_states = False
    _.priv_observe_contact_forces = False
    _.priv_observe_foot_displacement = False
    _.priv_observe_gravity_transformed_foot_displacement = False
    _.priv_observe_height_scan = True

    _.feasible_observe_command = True
    _.feasible_observe_height_scan = True

    _.num_privileged_obs = 2 + 187
    _.num_feasibility_obs = 15 + 187
    _.num_observation_history = 30

    _.observe_two_prev_actions = True
    _.observe_yaw = False
    _.num_observations = 70
    _.num_scalar_observations = 70
    _.observe_gait_commands = True
    _.observe_timing_parameter = False
    _.observe_clock_inputs = True

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30

    _.distributional_commands = True
    _.num_commands = 15
    _.resampling_time = 10
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1.0, 1.0]
    _.body_height_cmd = [-0.25, 0.15]
    _.gait_frequency_cmd_range = [2.0, 4.0]
    _.gait_phase_cmd_range = [0.0, 1.0]
    _.gait_offset_cmd_range = [0.0, 1.0]
    _.gait_bound_cmd_range = [0.0, 1.0]
    _.gait_duration_cmd_range = [0.5, 0.5]
    _.footswing_height_range = [0.03, 0.35]
    _.body_pitch_range = [-0.4, 0.4]
    _.body_roll_range = [-0.0, 0.0]
    _.stance_width_range = [0.10, 0.45]
    _.stance_length_range = [0.35, 0.45]

    _.limit_vel_x = [-5.0, 5.0]
    _.limit_vel_y = [-0.6, 0.6]
    _.limit_vel_yaw = [-5.0, 5.0]
    _.limit_body_height = [-0.25, 0.15]
    _.limit_gait_frequency = [2.0, 4.0]
    _.limit_gait_phase = [0.0, 1.0]
    _.limit_gait_offset = [0.0, 1.0]
    _.limit_gait_bound = [0.0, 1.0]
    _.limit_gait_duration = [0.5, 0.5]
    _.limit_footswing_height = [0.03, 0.35]
    _.limit_body_pitch = [-0.4, 0.4]
    _.limit_body_roll = [-0.0, 0.0]
    _.limit_stance_width = [0.10, 0.45]
    _.limit_stance_length = [0.35, 0.45]

    _.num_bins_vel_x = 21
    _.num_bins_vel_y = 1
    _.num_bins_vel_yaw = 21
    _.num_bins_body_height = 1
    _.num_bins_gait_frequency = 1
    _.num_bins_gait_phase = 1
    _.num_bins_gait_offset = 1
    _.num_bins_gait_bound = 1
    _.num_bins_gait_duration = 1
    _.num_bins_footswing_height = 1
    _.num_bins_body_roll = 1
    _.num_bins_body_pitch = 1
    _.num_bins_stance_width = 1
    
    _.exclusive_phase_offset = False
    _.pacing_offset = False
    _.binary_phases = True
    _.gaitwise_curricula = True

    _ = Cnfg.domain_rand
    _.rand_interval_s = 4
    _.randomize_base_mass = True
    _.added_mass_range = [-1.5, 1.5]
    _.push_robots = True
    _.max_push_vel_xy = 1.0
    _.randomize_friction = True
    _.friction_range = [0.1, 3.0]
    _.randomize_restitution = True
    _.restitution_range = [0.0, 0.4]
    _.restitution = 0.5  # default terrain restitution
    _.randomize_com_displacement = True
    _.com_displacement_range = [-0.15, 0.15]
    _.randomize_motor_strength = True
    _.motor_strength_range = [0.95, 1.05]
    _.randomize_kp_factor = True
    _.kp_factor_range = [-3.0, 3.0]
    _.randomize_kd_factor = True
    _.kd_factor_range = [-0.2, 0.2]
    _.randomize_lag_timesteps = True
    _.rand_interval_s = 6
    _.randomize_gravity = True
    _.gravity_range = [-1.0, 1.0]
    _.gravity_rand_interval_s = 8.0
    _.gravity_impulse_duration = 0.99
    _.randomize_rigids_after_start = False
    _.randomize_motor_offset = False
    _.motor_offset_range = [-0.02, 0.02]
 
    _ = Cnfg.curriculum_thresholds
    _.tracking_ang_vel = 0.7
    _.tracking_lin_vel = 0.8
    _.tracking_contacts_shaped_vel = 0.90
    _.tracking_contacts_shaped_force = 0.90

    _ = Cnfg.normalization
    _.friction_range = [0, 1]
    _.ground_friction_range = [0, 1]
    _.clip_actions = 10.0
