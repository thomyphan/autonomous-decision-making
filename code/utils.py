import pickle
from datetime import datetime
import os

def save_agent(agent, filename="agent"):
    os.makedirs("saved_agents", exist_ok=True)
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"saved_agents/{filename}: {date_string}.pkl", 'wb') as file:
        pickle.dump(agent, file)

def load_agent(path, verbose=False):
    with open(path, 'rb') as file:
        agent = pickle.load(file)
    if verbose: 
        agent.print()
    return agent