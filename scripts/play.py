import isaacgym

assert isaacgym
import glob
import pickle
import time

import numpy as np
import torch

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.utils.joystick_utils import JoystickManager


def real_time_sleep(start_time, target_dt):
    _end = time.perf_counter()
    elapsed = _end - start_time
    diff_to_target_period = target_dt - elapsed
    if diff_to_target_period > 0.0:
        time.sleep(diff_to_target_period)


def load_policy(logdir):
    body = torch.jit.load(logdir + "/checkpoints/wtw_policy.pt")
    feasibility = torch.jit.load(logdir + "/checkpoints/wtw_feasibility.pt")
    # adaptation_module = torch.jit.load(
    #     logdir + "/checkpoints/adaptation_module_latest.jit"
    # )

    def policy(obs, info={}):
        # latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        # action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        # info["latent"] = latent

        action = body.forward(obs["obs_history"].to("cpu"))
        return action
    
    def feasibility_net(obs, info={}):
        return feasibility.forward(obs["feasibility_obs"].to("cpu"))

    return policy, feasibility_net


def load_env(label, headless, joystick):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]
    print(f"Loading from {logdir}")

    with open(logdir + "/parameters.pkl", "rb") as file:
        pkl_cfg = pickle.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    # Cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2, 0, 0, 0, 0.0] # default
    Cfg.terrain.terrain_proportions = [0.3, 0.3, 0.0, 0.0, 0.4, 0, 0, 0, 0.0]
    # Cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1.0]
    
    Cfg.terrain.curriculum = False
    Cfg.terrain.measure_heights = True
    Cfg.rewards.use_terminal_body_height = False

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"
    Cfg.asset.terminate_after_contacts_on = []

    if joystick:
        Cfg.env.episode_length_s = 10000000000

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    policy, feasibility_net = load_policy(logdir)

    return env, policy, feasibility_net


def play_go1(headless, plot, joystick, real_time):
    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/2025-01-11/train"
    env, policy, feasibility_net = load_env(label, headless, joystick)

    if joystick:
        joystick_ctrl = JoystickManager(display=True)

    num_eval_steps = 250
    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5],
    }

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    i = 0
    while i < num_eval_steps:
        start_t = time.perf_counter()

        with torch.no_grad():
            actions = policy(obs)
            feasibility_pred = feasibility_net(obs)

        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd

        if joystick:
            joystick_ctrl.update()
            env.commands[:, 0] = -joystick_ctrl.left_xy[1].item()
            env.commands[:, 1] = -joystick_ctrl.left_xy[0].item()
            env.commands[:, 2] = -joystick_ctrl.right_xy[0].item() * 0.6
            env.commands[:, 5:8] = torch.tensor(gaits[joystick_ctrl.gait_name])

            joystick_ctrl.feasibility_value = feasibility_pred[0].item()
            joystick_ctrl.feasibility_gt = obs["feasibility_targets"][0].item()
        obs, rew, done, info = env.step(actions)

        if plot:
            measured_x_vels[i] = env.base_lin_vel[0, 0]
            joint_positions[i] = env.dof_pos[0, :].cpu()
            i += 1
        
        if real_time:
            real_time_sleep(start_t, env.dt)

    if not plot:
        return

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        measured_x_vels,
        color="black",
        linestyle="-",
        label="Measured",
    )
    axs[0].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        target_x_vels,
        color="black",
        linestyle="--",
        label="Desired",
    )
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        joint_positions,
        linestyle="-",
        label="Measured",
    )
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    play_go1(headless=False, plot=False, joystick=True, real_time=True)
