import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm

from model import ToyModel
from score import DoughnutScoring

class RLEnvironment:
    """Represents the environment for Reinforcement Learning with state and action management."""

    def __init__(self, c_min=0.05, c_max=0.95, eta_min=0.05, eta_max=0.95, num_c_states=10, num_eta_states=10, num_actions=9):
        """
        Initializes the RL environment with the given parameters.

        Args:
            c_min (float): Minimum value for c.
            c_max (float): Maximum value for c.
            eta_min (float): Minimum value for eta.
            eta_max (float): Maximum value for eta.
            num_c_states (int): Number of c states.
            num_eta_states (int): Number of eta states.
            num_actions (int): Number of possible actions.
        """

        self.c_min = c_min
        self.c_max = c_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.num_c_states = num_c_states
        self.num_eta_states = num_eta_states
        self.num_actions = num_actions

        self.c_values = np.linspace(self.c_min, self.c_max, self.num_c_states)
        self.c_step = (self.c_max - self.c_min) / (self.num_c_states - 1)
        self.eta_values = np.linspace(self.eta_min, self.eta_max, self.num_eta_states)
        self.eta_step = (self.eta_max - self.eta_min) / (self.num_eta_states - 1)

        self.central_states_c = range(1, self.num_c_states-1)
        self.central_states_eta = range(1, self.num_eta_states-1)

        self.action_meanings = {
            0: (-1, 1), 1: (0, 1), 2: (1, 1),
            3: (-1, 0), 4: (0, 0), 5: (1, 0),
            6: (-1, -1), 7: (0, -1), 8: (1, -1)
        }

    def find_possible_actions(self, c_state_init, eta_state_init, pop=False):
        """
        Determines possible actions based on the current state.

        Args:
            c_state_init (int): Initial state for c.
            eta_state_init (int): Initial state for eta.
            pop (bool): If True, removes action 4 from possible actions.

        Returns:
            np.ndarray: Array of possible actions.
        """
        if c_state_init == 0 and eta_state_init == 0:
            possible_actions = [1, 2, 4, 5]

        elif c_state_init in self.central_states_c and eta_state_init == 0:
            possible_actions = [0, 1, 2, 3, 4, 5]

        elif c_state_init == self.num_c_states-1 and eta_state_init == 0:
            possible_actions = [0, 1, 3, 4]

        elif c_state_init == self.num_c_states-1 and eta_state_init in self.central_states_eta:
            possible_actions = [0, 1, 3, 4, 6, 7]

        elif c_state_init == self.num_c_states-1 and eta_state_init == self.num_eta_states-1:
            possible_actions = [3, 4, 6, 7]

        elif c_state_init in self.central_states_c and eta_state_init == self.num_eta_states-1:
            possible_actions = [3, 4, 5, 6, 7, 8]

        elif c_state_init == 0 and eta_state_init == self.num_eta_states-1:
            possible_actions = [4, 5, 7, 8]

        elif c_state_init == 0 and eta_state_init in self.central_states_eta:
            possible_actions = [1, 2, 4, 5, 7, 8]

        elif c_state_init in self.central_states_c and eta_state_init in self.central_states_eta:
            possible_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        if pop:
            possible_actions.remove(4) if 4 in possible_actions else None

        return np.array(possible_actions).astype(int)

    def find_rewardable_actions(self, rewardable_actions, c_change_current, eta_change_current):
        """
        Filters out actions that contradict the current changes in c and eta.

        Args:
            rewardable_actions (list): List of actions to check.
            c_change_current (int): Current change in c.
            eta_change_current (int): Current change in eta.

        Returns:
            np.ndarray: Array of valid rewardable actions.
        """
        possible_actions = []

        for action in rewardable_actions:
            c_change, eta_change = self.action_meanings[action]

            if c_change_current != 0 and eta_change_current != 0:
                if (c_change != -c_change_current) and (eta_change != -eta_change_current):
                    possible_actions.append(action)
            elif c_change_current != 0 and eta_change_current == 0:
                if c_change != -c_change_current:
                    possible_actions.append(action)
            elif c_change_current == 0 and eta_change_current != 0:
                if eta_change != -eta_change_current:
                    possible_actions.append(action)
            elif c_change_current == 0 and eta_change_current == 0:
                possible_actions.append(action)

        return np.array(possible_actions)


class RLAgent:
    """Represents an agent that interacts with the RL environment."""

    def __init__(self, env, alpha_lr=0.1, gamma=0.5, inv_temperature=2, save=True):
        """
        Initializes the RL agent.

        Args:
            env (RLEnvironment): The RL environment to interact with.
            alpha_lr (float): Learning rate for Q-value update.
            gamma (float): Discount factor.
            inv_temperature (float): Inverse temperature for the softmax function.
        """
        self.env = env
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.inv_temperature = inv_temperature

        self.Q = np.zeros((env.num_c_states, env.num_eta_states, env.num_actions))
        self.V = np.zeros((env.num_c_states, env.num_eta_states))
        self.R = np.zeros((env.num_c_states, env.num_eta_states))
        
        self.log_params() if save else None

    def log_params(self):
        
        self.timestamp = self.generate_timestamp()
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'rl')
        os.makedirs(data_dir, exist_ok=True)
    
        log_path = os.path.join(data_dir, f'rl_agent_params_{self.timestamp}.json')
        params = {
            "alpha_lr": self.alpha_lr,
            "self.gamma": self.gamma,
            "self.inv_temperature": self.inv_temperature
        }
        with open(log_path, 'w') as f:
            json.dump(params, f, indent=2)

    # Helper function to generate timestamp
    def generate_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def softmax(self, Q_values, inv_temperature):
        """
        Computes the softmax of the Q-values.

        Args:
            Q_values (np.ndarray): Array of Q-values.

        Returns:
            np.ndarray: Array of action probabilities.
        """
        exp_values = np.exp(Q_values * inv_temperature)
        return exp_values / np.sum(exp_values)

    def update_Q(self, c_state, eta_state, action, reward):
        """
        Updates the Q-values using the Q-learning update rule.

        Args:
            c_state (int): Current state for c.
            eta_state (int): Current state for eta.
            action (int): Action taken.
            reward (float): Reward received.

        Returns:
            np.ndarray: Updated Q-values.
        """
        self.Q[c_state, eta_state, action] += self.alpha_lr * (reward - self.Q[c_state, eta_state, action])
        return self.Q


class RLTrainer:
    """Handles the training loop for the RL agent."""

    def __init__(self, agent, scorer, num_episodes=20000, episode_steps=50):
        """
        Initializes the RL trainer.

        Args:
            agent (RLAgent): The agent to train.
            num_episodes (int): Number of training episodes.
            episode_steps (int): Number of steps per episode.
        """
        self.agent = agent
        self.num_episodes = num_episodes
        self.episode_steps = episode_steps

        self.Q_time = np.zeros((num_episodes, episode_steps, self.agent.env.num_c_states, self.agent.env.num_eta_states, self.agent.env.num_actions))
        
        self.set_rewards(scorer)
        self.set_folders()

    def set_rewards(self, scorer):

        self.R = self.compute_reward_matrix(scorer)

        c_state_ref = int( np.argmax(self.R, axis=0).mean() )
        eta_state_ref = np.argmax(self.R, axis=1)[0]
        self.sref = np.array( [ c_state_ref, eta_state_ref ] ) # best state

    def set_folders(self):

        self.timestamp = self.agent.timestamp

        RL_folder = 'rl'
        os.makedirs(RL_folder, exist_ok=True)
        self.rl_data_path = os.path.join(os.path.dirname(__file__), 'data', RL_folder, f'run_{self.timestamp}')
        os.makedirs(self.rl_data_path, exist_ok=True)

    def compute_reward_matrix(self, scorer):
        """
        Computes the reward matrix R by solving the system dynamics for each combination of
        c_state and eta_state using the model, and scoring the resulting trajectory.
        """
        
        state_index = pd.MultiIndex.from_product([self.agent.env.c_values, self.agent.env.eta_values], names=['c', 'eta'])
        R = np.array([scorer.score_model(params) for params in state_index])            
        R = np.reshape(R, (self.agent.env.num_c_states, self.agent.env.num_eta_states))
        
        return R

    def train(self):
        """Trains the RL agent for the specified number of episodes."""

        [self.run_episode() for episode in tqdm(range(self.num_episodes))]

    def run_episode(self):

        # Initialize state
        #c_state_0, eta_state_0 = 8, 1 #fix initial position (comment out following line too)
        c_state_0, eta_state_0 = np.random.randint(1, self.agent.env.num_eta_states-1), np.random.randint(1, self.agent.env.num_eta_states-1)
    
        c_state_init, eta_state_init = c_state_0, eta_state_0
        possible_actions = self.agent.env.find_possible_actions(c_state_init, eta_state_init)

        for t in range(self.episode_steps):
            Q = self.agent.Q[c_state_init, eta_state_init, possible_actions]
            p = self.agent.softmax(Q, self.agent.inv_temperature) #action probabilities
            action = np.random.choice(possible_actions, p=p)

            # Update state
            c_change, eta_change = self.agent.env.action_meanings[action]
            c_state = max(0, min(self.agent.env.num_c_states - 1, c_state_init + c_change))
            eta_state = max(0, min(self.agent.env.num_eta_states - 1, eta_state_init + eta_change))

            reward = self.R[c_state, eta_state]
            possible_actions_next_state = self.agent.env.find_possible_actions(c_state, eta_state)
            reward += self.agent.gamma * np.max(self.agent.Q[c_state, eta_state, possible_actions_next_state])

            # Update Q-values
            self.agent.update_Q(c_state_init, eta_state_init, action, reward)

            c_state_init, eta_state_init = c_state, eta_state
            possible_actions = possible_actions_next_state

    def save(self, time=False):
        """Saves necessary files after training."""

        np.save(self.rl_data_path + '/sref', self.sref) 
        np.save(self.rl_data_path + '/Q', self.agent.Q)
        np.save(self.rl_data_path + '/V', self.agent.V)
        np.save(self.rl_data_path + '/R', self.R)
        np.save(self.rl_data_path + '/Q_time', self.Q_time) if time else None
        
# Running the script
def main():
    rl_env = RLEnvironment()
    rl_agent = RLAgent(env=rl_env)
    
    model = ToyModel()
    scorer = DoughnutScoring(model)

    rl_trainer = RLTrainer(agent=rl_agent, scorer=scorer)
    rl_trainer.train()
    rl_trainer.save()

if __name__ == "__main__":
    main()
