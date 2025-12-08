import argparse
import os

from envs.car_wrapper import make_car_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_vec_env(seed: int):
    """
    Factory that returns a DummyVecEnc around our CarRacing env.
    """
    
    def _make():
        env = make_car_env(seed=seed)
        env = Monitor(env)
        return env
    
    return DummyVecEnv([_make])

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--steps", type=int, default=200_000, help="Total timesteps to train PPO")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment and PPO")
    parser.add_argument("--logdir", type=str, default="runs\ppo", help="TensorBoard log directory")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.logdir, exist_ok=True)
    
    vec_env = make_vec_env(seed=args.seed)
    
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )
    
    model.learn(total_timesteps=args.steps, tb_log_name=f"ppo_seed{args.seed}")

    save_path = os.path.join(args.logdir, f"ppo_seed{args.seed}.zip")
    model.save(save_path)
    vec_env.close()
    print(f"Training finished. Model saved to {save_path}")
    
if __name__ == "__main__":
    main()