import rooms
import agent as a
import matplotlib.pyplot as plot
import sys

def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
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
    
def print_info(agent, params, training_episodes, rooms_instance):
    print("Agent:", agent.__class__.__name__)
    print("Environment:", rooms_instance)
    print("Discount factor (gamma):", params["gamma"])
    print("Learning rate (alpha):", params["alpha"])
    print("Exploration constant:", params["exploration_constant"])
    print("Epsilon decay:", params["epsilon_decay"])
    print("Training for", training_episodes, "episodes.")
    print("")

params = {}
rooms_instance = sys.argv[1]
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.0001
params["alpha"] = 0.1
params["env"] = env
params["exploration_constant"] = 5

#agent = a.RandomAgent(params)
#agent = a.SARSALearner(params)
agent = a.QLearner(params)
# agent = a.UCBQLearner(params)
training_episodes = 2000
print_info(agent, params, training_episodes, rooms_instance)

returns = [episode(env, agent, i) for i in range(training_episodes)]

assert params['gamma'] == 0.99
good_returns = []
prev_return = []
for i in range(len(returns)):
    if returns[i] >= 0.8:
        prev_return.append(returns[i])
    else:
        if len(prev_return) >= 10:
            good_returns.append((i - len(prev_return), prev_return))
        prev_return = []

if len(prev_return) >= 10:
    good_returns.append((len(returns) - len(prev_return), prev_return))

for i in good_returns:
    print("Start:", i[0], "Length:", len(i[1]), "Returns:", i[1])

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.show()
plot.savefig(f"{rooms_instance}.png")

env.save_video()
