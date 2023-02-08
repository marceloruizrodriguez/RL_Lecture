from stable_baselines3 import PPO
from maintEnv import FactoryEnv
import wandb
from wandb.integration.sb3 import WandbCallback

env = FactoryEnv("./Config/env.json","config_2")

# config = {
#     "policy_type": "MlpPolicy",
#     "total_timesteps": 1000000
# }
# wandb.login(key="")
# run = wandb.init(
#     project="RLGA",
#     entity="marceloruiz",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     # monitor_gym=True,  # auto-upload the videos of agents playing the game
#     # save_code=True,  # optional
# )
#
# model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
# model.learn(
#     total_timesteps=config["total_timesteps"],
#     callback=WandbCallback(
#         model_save_path=f"models/{run.id}",
#         verbose=2,
#     ),
# )
# run.finish()