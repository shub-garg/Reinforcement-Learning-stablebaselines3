import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

environment_name = "CartPole-v1"
env = gym.make(environment_name, render_mode="rgb_array")

episodes = 5

for episode in range(1, episodes+1):
    state, info = env.reset()
    terminated, truncated = False, False
    score = 0

    while not (terminated or truncated):
        frame = env.render()
        action = env.action_space.sample()  # Sample a random action
        n_state, reward, terminated, truncated, info = env.step(action)  # Correctly unpack the return values
        score += reward  # Accumulate the reward

        # Convert frame to BGR format for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('CartPole-v1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #time.sleep(0.4)  # Add delay for rendering

    print('Episode:{} Score:{}'.format(episode, score))  # Print the total score for the episode

env.close()
cv2.destroyAllWindows()

save_path = os.path.join(r"training","Saved Models")

log_path = os.path.join(r'training','logs')

model = DQN('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

stop_call = StopTrainingOnRewardThreshold(reward_threshold=500,verbose=1)
eval_call = EvalCallback(env, callback_on_new_best=stop_call, eval_freq=10000, best_model_save_path=save_path, verbose=1)
model.learn(total_timesteps=200000, callback=eval_call)

DQN_path = os.path.join(r"training","Saved Models","DQN_Model_Catpole")

model.save(DQN_path)

episodes = 20

for episode in range(1, episodes + 1):
    obs = env.reset()  # Reset environment and get the initial observation
    done = False
    score = 0

    while not done:
        frame = env.render()
        action, _ = model.predict(obs)  # Predict action using the model
        obs, reward, done, info = env.step(action)  # Step the environment
        score += reward  # Accumulate the reward

        # Convert frame to BGR format for OpenCV
        if frame is not None:  # Check if the frame is not empty
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('CartPole-v1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print('Episode:{} Score:{}'.format(episode, score))  # Print the total score for the episode

env.close()
cv2.destroyAllWindows()