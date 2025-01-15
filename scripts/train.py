def train_go1(headless=True):
    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from go1_gym_learn.ppo_cse import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import RunnerArgs

    config_go1(Cfg)
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    # log the experiment parameters
    logger.log_params(
        AC_Args=vars(AC_Args),
        PPO_Args=vars(PPO_Args),
        RunnerArgs=vars(RunnerArgs),
        Cfg=vars(Cfg),
    )

    env = HistoryWrapper(env)
    runner = Runner(env, device=f"cuda:0")
    runner.learn(num_learning_iterations=30000, init_at_random_ep_len=True)


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(
        logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
        root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), 
    )

    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/terrain_level/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: feasibility_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_go1(headless=True)
