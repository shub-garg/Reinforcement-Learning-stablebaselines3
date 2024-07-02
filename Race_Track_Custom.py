import os
import gym
import cv2
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Set environment name
environment_name = "CarRacing-v2"

# Setup directories
log_path = os.path.join("car_racing_training", "logs")
save_path = os.path.join("car_racing_training", "Saved Models")
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# Create environment with rendering mode "human" (for video) and "rgb_array" (for training)
env = gym.make(environment_name, render_mode="rgb_array")

# Training parameters
total_timesteps = 200000
eval_freq = 10000
eval_episodes = 10

# Define ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define custom CNN policy with ResNet-like structure
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.resnet = nn.Sequential(
            ResNetBlock(n_input_channels, 32),
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(256, features_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Create a DummyVecEnv for stable baselines
env = DummyVecEnv([lambda: env])

# Define custom policy with the custom CNN feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256)
)

# Initialize PPO model with custom policy
model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path)

# Define callbacks
stop_call = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
eval_call = EvalCallback(env, callback_on_new_best=stop_call, eval_freq=eval_freq,
                         best_model_save_path=save_path, verbose=1)

# Train the model
model.learn(total_timesteps=total_timesteps, callback=eval_call)

# Save the trained model
model_path = os.path.join(save_path, "PPO_Model_CarRacing")
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
            cv2.imshow('CarRacing-v2', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        action, _ = model.predict(obs)  # Predict action using the model
        obs, reward, done, info = env.step(action)  # Step the environment
        score += reward  # Accumulate the reward

    print('Episode:{} Score:{}'.format(episode, score))  # Print the total score for the episode

# Close OpenCV windows and environment
cv2.destroyAllWindows()
env.close()
