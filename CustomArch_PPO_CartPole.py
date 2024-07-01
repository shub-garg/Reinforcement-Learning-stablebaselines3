import os
import gymnasium as gym
from stable_baselines3 import PPO
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
        n_state, reward, terminated, truncated, info = env.step(action) 
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


log_path = os.path.join(r'training','logs')

env = gym.make(environment_name, render_mode="rgb_array")
env = DummyVecEnv([lambda: env])

stop_call = StopTrainingOnRewardThreshold(reward_threshold=500,verbose=1)

save_path = os.path.join(r"training","Saved Models")

eval_call = EvalCallback(env, callback_on_new_best=stop_call, eval_freq=10000, best_model_save_path=save_path, verbose=1)

net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])]

model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch})
model.learn(total_timesteps=20000, callback=eval_call)

PPO_path = os.path.join(r"training","Saved Models","PPO_Model_Catpole")

model.save(PPO_path)

model = PPO.load(PPO_path,env=env)

res = evaluate_policy(model,env,n_eval_episodes=10,render=True)

print("Score is:", res)

env.close()

episodes = 5

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