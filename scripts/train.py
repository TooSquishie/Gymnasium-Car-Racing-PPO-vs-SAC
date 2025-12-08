import argparse
import os

from envs.car_wrapper import make_car_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_vec_env(seed: int):
    def _make():
        env = make_car_env(seed=seed)
        env = Monitor(env)
        return env
    
    return DummyVecEnv([_make])


def train_once(algo: str, steps: int, logdir: str, seed: int):
    os.makedirs(logdir, exist_ok=True)

    vec_env = make_vec_env(seed)

    if algo == "ppo":
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=logdir,
            seed=seed,
        )
    elif algo == "sac":
        model = SAC(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=logdir,
            seed=seed,
        )
    else:
        raise ValueError("algo must be 'ppo' or 'sac'")

    model.learn(total_timesteps=steps, tb_log_name=f"{algo}_seed{seed}")

    save_path = os.path.join(logdir, f"{algo}_seed{seed}.zip")
    model.save(save_path)
    
    vec_env.close()
    print(f"[{algo.upper()}] seed={seed} finished. Saved to {save_path}")


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--algo", choices=["ppo", "sac"], required=True, help="Which RL algorithm to train")
    p.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps per seed")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of seeds, e.g. '0,1,2,3,4'")
    p.add_argument("--logdir", type=str, default="runs", help="Base directory for logs and checkpoints")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # logdir/ppo or logdir/sac
    algo_logdir = os.path.join(args.logdir, args.algo)

    for s in seeds:
        train_once(
            algo=args.algo,
            steps=args.steps,
            logdir=algo_logdir,
            seed=s,
        )
