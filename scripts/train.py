import argparse
import os

from envs.car_wrapper import make_car_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_vec_env(seed: int, hardmode: bool):
    def _make():
        env = make_car_env(seed=seed, render_mode=None, hard_mode=hardmode)
        env = Monitor(env)
        return env
    
    return DummyVecEnv([_make])

def build_model(algo: str, vec_env, logdir: str, seed: int):
    if algo == "ppo":
        # On-policy, more stable but less sample-efficient
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=logdir,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,         # rollout length
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=4,
            max_grad_norm=0.5,
        )
    elif algo == "sac":
        # Off-policy, sample-efficient, good for continuous control
        model = SAC(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=logdir,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            learning_starts=20_000,
            ent_coef="auto",
        )
    else:
        raise ValueError("algo must be 'ppo' or 'sac'")

    return model

def train_once(algo: str, steps: int, base_logdir: str, seed: int, hard_mode: bool):
    """
    Train a single model (algo, seed, mode) and save checkpoint + TensorBoard logs.
    """
    # e.g. runs/ppo or runs/sac
    algo_logdir = os.path.join(base_logdir, algo)
    os.makedirs(algo_logdir, exist_ok=True)

    vec_env = make_vec_env(seed=seed, hard_mode=hard_mode)
    model = build_model(algo, vec_env, algo_logdir, seed)

    tb_name = f"{algo}_seed{seed}" + ("_hard" if hard_mode else "")
    model.learn(total_timesteps=steps, tb_log_name=tb_name)

    ckpt_name = f"{algo}_seed{seed}"
    if hard_mode:
        ckpt_name += "_hard"
    ckpt_path = os.path.join(algo_logdir, f"{ckpt_name}.zip")

    model.save(ckpt_path)
    vec_env.close()

    print(f"[{algo.upper()}] seed={seed} hard_mode={hard_mode} finished.")
    print(f"  Saved model to: {ckpt_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True, help="Which RL algorithm to train")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps per seed")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of seeds, e.g. '0,1,2,3,4'")
    parser.add_argument("--logdir", type=str, default="runs", help="Base directory for TensorBoard logs and checkpoints")
    parser.add_argument("--hard-mode", action="store_true", help="Enable noisy/delayed control and early off-track termination")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    for s in seeds:
        train_once(
            algo=args.algo,
            steps=args.steps,
            base_logdir=args.logdir,
            seed=s,
            hard_mode=args.hard_mode,
        )
