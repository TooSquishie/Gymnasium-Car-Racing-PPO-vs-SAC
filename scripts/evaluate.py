import argparse

from envs.car_wrapper import make_car_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True, help="Which algorithm the checkpoint belongs to.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .zip model file (e.g. runs/ppo/ppo_seed0.zip)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for held-out evaluation tracks.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    env = make_car_env(seed=args.seed, render_mode=None)
    
    if args.algo == "ppo":
        model = PPO.load(args.checkpoint)
    else:
        model = SAC.load(args.checkpoint)
        
    mean_r, std_r = evaluate_policy(
    model,
    env,
    n_eval_episodes=args.episodes,
    deterministic=True,
    )
    
    print(f"[{args.algo.upper()}] Evaluation on seed={args.seed}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Mean reward:   {mean_r:.2f}")
    print(f"  Std reward:    {std_r:.2f}")

    env.close()


if __name__ == "__main__":
    main()