import os
from pathlib import Path
# from typing import Callable

# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from lib.Callbacks import TensorboardCallback
from MotionEnv import MotionEnv

# models_dir = os.path.join(os.path.dirname(
#     __file__), '..', 'models', env_name + '-' + algorithm_name)
# logdir = os.path.join(os.path.dirname(
#     __file__), '..', 'logs', env_name + '-' + algorithm_name)

# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)

# if not os.path.exists(logdir):
#     os.makedirs(logdir)


def train_agent(env, algorithm, params={}):

    env_name = env.__class__.__name__
    algorithm_name = algorithm.__name__

    models_dir = os.path.join(os.path.dirname(
        __file__), 'models', env_name + '-' + algorithm_name)
    logdir = os.path.join(os.path.dirname(
        __file__), 'logs', env_name + '-' + algorithm_name)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    paths = sorted(Path(models_dir).iterdir(), key=os.path.getmtime)

    last_model = None
    last_iter = 0

    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])
    env.reset()

    if len(paths) > 0:
        # get last model file
        last_model = paths[-1]

        # get last iteration
        last_iter = int(os.path.splitext(last_model.name)[0])

        last_model = algorithm.load(last_model, env, verbose=1,
                                    tensorboard_log=logdir, **params)

    if last_model:
        model = last_model
    else:
        # model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)
        model = algorithm('MlpPolicy', env, verbose=1,
                          tensorboard_log=logdir, **params)

    TIMESTEPS = 100000

    tensorboard_callback = TensorboardCallback()
    # # Create the callback object
    # eval_callback = EvalCallback(eval_env=env, best_model_save_path=models_dir,
    #                              log_path=logdir, eval_freq=1000,
    #                              deterministic=True, render=False)

    # with ProgressBarManager(TIMESTEPS) as progress_callback:
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=f"{last_iter+TIMESTEPS}",
                callback=[tensorboard_callback])

    model.save(f"{models_dir}/{last_iter + TIMESTEPS}.zip")


def check_model():

    env = MotionEnv()
    env.reset()

    # models_dir = os.path.join(os.path.dirname(
    #     __file__), 'models', env.__class__.__name__ + '-' + 'PPO')

    # paths = sorted(Path(models_dir).iterdir(), key=os.path.getmtime)

    # model = PPO.load(paths[-1], env=env)

    model = PPO('MlpPolicy', env, verbose=1)

    # print(dir(model))
    print(model.policy)
    """
    ActorCriticPolicy(
        (features_extractor): FlattenExtractor(
            (flatten): Flatten(start_dim=1, end_dim=-1)
        )
        (pi_features_extractor): FlattenExtractor(
            (flatten): Flatten(start_dim=1, end_dim=-1)
        )
        (vf_features_extractor): FlattenExtractor(
            (flatten): Flatten(start_dim=1, end_dim=-1)
        )
        (mlp_extractor): MlpExtractor(
            (policy_net): Sequential(
            (0): Linear(in_features=112, out_features=64, bias=True)
            (1): Tanh()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): Tanh()
            )
            (value_net): Sequential(
            (0): Linear(in_features=112, out_features=64, bias=True)
            (1): Tanh()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): Tanh()
            )
        )
        (action_net): Linear(in_features=64, out_features=56, bias=True)
        (value_net): Linear(in_features=64, out_features=1, bias=True)
    )
    """

    # print(model.policy.features_extractor)
    # print(model.policy.mlp_extractor)
    # print(model.policy_kwargs)


if __name__ == "__main__":

    # check_model()

    env = MotionEnv()
    # env.reset()

    train_agent(env, PPO, params={})
