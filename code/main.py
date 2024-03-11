import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
from utils import save_agent, load_agent
import numpy as np

def episode(env, agent, discount_factor = 0.99, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return


params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["env"] = env

np.random.seed(42)

#agent = a.RandomAgent(params)
#agent = a.SARSALearner(params)
agent = a.QLearner(params)
# agent = load_agent("saved_agents/agent: 2024-03-11 13:06:17.pkl")
training_episodes = 200
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.show()

# env.save_video()
