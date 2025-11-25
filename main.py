# %%
import gymnasium as gym
from agent.agent import root_agent
from agent.coach_callback import LLMCoachCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# %%
env_id = "LunarLander-v3"
env_desc = """
## Description
This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

## Action Space
There are four discrete actions available:

- 0: do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine

## Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear velocities in `x` & `y`, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Starting State
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

## Episode Termination
The episode finishes if:
1. the lander crashes (the lander body gets in contact with the moon);
2. the lander gets outside of the viewport (x coordinate is greater than 1);
3. the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:
"""

n_envs = 16

# %%
env = gym.make(env_id, render_mode="rgb_array")
vec_env = make_vec_env(env_id, n_envs=n_envs)
model = DQN("MlpPolicy", env, verbose=0)

# %%
callback = LLMCoachCallback(agent=root_agent, env_desc=env_desc)
model.learn(total_timesteps=250_000, progress_bar=True, callback=callback)
model.save("dqn_model")

# %%
eval_env = gym.make(env_id, render_mode="human")
obs, _ = eval_env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)

    obs, rewards, terminated, truncated, info = eval_env.step(action.item())

    if terminated or truncated:
        obs, _ = eval_env.reset()
