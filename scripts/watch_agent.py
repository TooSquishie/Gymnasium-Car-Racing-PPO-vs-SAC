import argparse
from envs.car_wrapper import make_car_env
from stable_baselines3 import PPO, SAC

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True, help="Which algorithm the checkpoint belongs to.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .zip model file (e.g. runs/ppo/ppo_seed0.zip)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to watch.")
    parser.add_argument("--seed", type=int, default=12345, help="Env seed (for reproducible tracks).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    env = make_car_env(seed=args.seed, render_mode="human")
    
    if args.algo == "ppo":
        model = PPO.load(args.checkpoint)
    else:
        model = SAC.load(args.checkpoint)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            # model expects (batch, C, H, W), but SB3 handles that inside predict()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        print(f"Episode {ep+1}: total reward = {ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()