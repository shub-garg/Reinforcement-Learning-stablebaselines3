import os
import gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Set environment name
environment_name = "Humanoid-v4"

# Setup directories
log_path = os.path.join("training", "logs")
save_path = os.path.join("training", "Saved Models")
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# Create environment with rendering mode "human" (for video) and "rgb_array" (for training)
env = gym.make(environment_name, render_mode="rgb_array")

# Training parameters
total_timesteps = 200000
eval_freq = 10000
eval_episodes = 10

# Define network architecture for policy and value function
net_arch = [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]

# Define callbacks
stop_call = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
eval_call = EvalCallback(env, callback_on_new_best=stop_call, eval_freq=eval_freq,
                         best_model_save_path=save_path, verbose=1)

# Create a DummyVecEnv for stable baselines
env = DummyVecEnv([lambda: env])

# Initialize PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch': net_arch})

# Train the model
model.learn(total_timesteps=total_timesteps, callback=eval_call)

# Save the trained model
model_path = os.path.join(save_path, "PPO_Model_Humanoid")
model.save(model_path)

# Load the trained model for evaluation
model = PPO.load(model_path, env=env)

# Evaluate the trained model
res = evaluate_policy(model, env, n_eval_episodes=eval_episodes, render=False)

# Print evaluation results
print("Evaluation Score:", res)

# Close the environment
env.close()

# Video rendering and testing
episodes = 5

for episode in range(1, episodes + 1):
    obs = env.reset()  # Reset environment and get the initial observation
    done = False
    score = 0

    while not done:
        frame = env.render(mode='rgb_array')  # Render the environment frame

        # Convert frame to BGR format for OpenCV display
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Humanoid-v4', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        action, _ = model.predict(obs)  # Predict action using the model
        obs, reward, done, info = env.step(action)  # Step the environment
        score += reward  # Accumulate the reward

    print('Episode:{} Score:{}'.format(episode, score))  # Print the total score for the episode

# Close OpenCV windows and environment
cv2.destroyAllWindows()
env.close()
