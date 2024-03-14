import rooms
import random
import agent as a
import matplotlib.pyplot as plot
import sys
from utils import save_agent, load_agent
import numpy as np

def plot_returns(x,y):
    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("Episode")
    plot.ylabel("Discounted Return")
    plot.show()


def episode(env, agent, discount_factor = 0.99, nr_episode=0, evaluation_mode=False, verbose=True):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    if evaluation_mode:
        agent.epsilon = 0
        agent.exploration_constant = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        if not evaluation_mode: 
            agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    if verbose: print(nr_episode, ":", discounted_return, "steps: ", time_step)
    return discounted_return


params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.0001
params["alpha"] = 0.1
params["env"] = env
params["exploration_constant"] = 5
params["epsilon"] = 1

np.random.seed(42)
random.seed(42)

#agent = a.RandomAgent(params)
# agent = a.SARSALearner(params)
agent = a.QLearner(params)

# agent = load_agent("saved_agents/agent: 2024-03-11 19:10:22.pkl") # Load agent from file
training_episodes = 2000
evaluation_episodes = 10

# TRAINING
returns = [episode(env, agent, i, verbose=False) for i in range(training_episodes)]
x = range(training_episodes)
y = returns
# plot_returns(x,y)

# EVALUATION
eval_returns = [episode(env, agent, i, verbose=True, evaluation_mode=True) for i in range(evaluation_episodes)]
x_eval = range(evaluation_episodes)
y_eval = eval_returns
# plot_returns(x_eval,y_eval)

print(f"Evaluation discounted reward: {np.mean(eval_returns)}")
# save_agent(agent)

# env.save_video()
