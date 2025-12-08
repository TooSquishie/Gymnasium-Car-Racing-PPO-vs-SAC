import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_environment():
    environment = gym.make("CarRacing-v3", domain_randomize = True, continuous = True)
    environment = Monitor(environment)
    return environment

def main():
    vec_environment = DummyVecEnv([make_environment])
    
    model = PPO(
        "CnnPolicy",
        vec_environment,
        verbose=1,
        tensorboard_log="runs/basic_ppo"
    )
    
    model.learn(total_timesteps=50_000)
    
    model.save("ppo_basic_car_training.zip")
    vec_environment.close()
    print("Training finished and model saved")
    
if __name__ == "__main__":
    main()