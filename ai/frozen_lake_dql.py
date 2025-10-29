import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn.functional as F
from torch import nn


# Define a Deep Q-Network (DQN) using PyTorch's nn.Module
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        # First fully connected layer: input -> hidden layer
        self.fc1 = nn.Linear(in_states, h1_nodes)

        # Output layer: hidden layer -> output (Q-values for each action)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        # Apply ReLU activation to the output of the first layer
        x = F.relu(self.fc1(x))

        # Output the Q-values for each action (no activation function here)
        x = self.out(x)

        return x

# ReplayMemory is a buffer to store past transitions (experiences)
class ReplayMemory():
    def __init__(self, maxlen):
        # Create a deque with a maximum length to store transitions
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        # Add a transition to the memory
        # A transition is usually a tuple: (state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, sample_size):
        # Randomly sample a batch of transitions from memory
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Return the current number of transitions in memory
        return len(self.memory)


class FrozenLakeDQL():
    # Hyperparameters
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32
    loss_fn = nn.MSELoss()
    optimizer = None
    ACTIONS = ['L', 'D', 'R', 'U']

    def train(self, episodes, render=False, is_slippery=False):
        # Create the FrozenLake environment
        env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=is_slippery,
                       render_mode='human' if render else None)

        num_states = env.observation_space.n
        num_actions = env.action_space.n
        epsilon = 1  # Starting value for epsilon-greedy
        memory = ReplayMemory(self.replay_memory_size)

        # Initialize policy and target DQNs
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())  # Sync weights initially

        print('Policy (random, before training): ')
        self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]  # Get initial state
            terminated = False
            truncated = False

            while not terminated and not truncated:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = self.state_to_dqn_input(state, num_states)
                        action = policy_dqn(state_tensor).argmax().item()

                # Take action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Store transition
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

            # Track successful episodes
            if reward == 1:
                rewards_per_episode[i] = 1

            # Start training once there's enough memory
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Sync target network
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()

        # Plotting performance
        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)
        plt.plot(sum_rewards)
        plt.title("Success Rate (100 ep window)")
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.savefig('frozen_lake_dqn.png')

        # Save model
        torch.save(policy_dqn.state_dict(), 'frozen_lake_dqn.pt')

    def print_dqn(self, policy_dqn):
        # Optional: print Q-values for each state
        with torch.no_grad():
            for state in range(policy_dqn.fc1.in_features):
                input_tensor = self.state_to_dqn_input(state, policy_dqn.fc1.in_features)
                q_values = policy_dqn(input_tensor)
                print(f"State {state}: {q_values.numpy()}")

    def state_to_dqn_input(self, state: int, num_states: int):
        # One-hot encode the state
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        num_states = policy_dqn.fc1.in_features
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            # Convert to tensors
            state_tensor = self.state_to_dqn_input(state, num_states)
            next_state_tensor = self.state_to_dqn_input(new_state, num_states)

            # Compute target Q value
            with torch.no_grad():
                if terminated:
                    target_value = torch.tensor(reward)
                else:
                    next_q_values = target_dqn(next_state_tensor)
                    target_value = torch.tensor(reward + self.discount_factor_g * next_q_values.max())

            # Compute current Q-values from policy DQN
            current_q_values = policy_dqn(state_tensor)
            current_q_list.append(current_q_values)

            # Clone and update the specific action value
            target_q_values = current_q_values.clone().detach()
            target_q_values[action] = target_value
            target_q_list.append(target_q_values)

        # Compute loss and optimize
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes, is_slippery=False):
        # Create the environment with visual rendering
        env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=is_slippery, render_mode='human')

        # Get environment parameters
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load the trained DQN model
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load('frozen_lake_dqn.pt'))
        policy_dqn.eval()  # Set to evaluation mode (disables dropout, etc.)

        print("Policy (trained): ")
        self.print_dqn(policy_dqn)

        success_count = 0  # Count successful episodes

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                with torch.no_grad():
                    # Select the best action based on current policy
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)

            # Track success (reward = 1 means goal reached)
            if reward == 1:
                success_count += 1

        env.close()

        # Print success rate after testing
        print(f"\nTest episodes: {episodes}")
        print(f"Successes: {success_count}")
        print(f"Success Rate: {success_count / episodes * 100:.2f}%")


if __name__=='__main__':
    frozen_lake=FrozenLakeDQL()
    is_slippery=False
    frozen_lake.train(1000,is_slippery=is_slippery)
    frozen_lake.test(4,is_slippery=is_slippery)