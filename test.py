import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset()

done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print("Total reward:", total_reward)
