############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import collections

import numpy as np
import torch

# Create a Network class, which inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class DQN:
    def __init__(self, lr):
        self.q_network = Network(input_dimension=2, output_dimension=3)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr)
        self.q_target_network = Network(input_dimension=2, output_dimension=3)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, batch, gamma):
        self.gamma = gamma

        self.optimiser.zero_grad()
        loss, sample_specific_losses = self._calculate_loss(batch)
        loss.backward()
        self.optimiser.step()

        return loss.item(), sample_specific_losses

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, batch):
        state, action, reward, next_state = batch

        prediction = self.q_network.forward(torch.tensor(state)).gather(1, torch.tensor(action))

        prediction_next_state = self.q_target_network.forward(torch.tensor(next_state)).detach().max(1)[0].unsqueeze(1)

        # Bellman equation
        labels = (torch.tensor(reward) + self.gamma * prediction_next_state)

        sample_specific_losses = abs(prediction - labels)
        loss = torch.nn.MSELoss()(prediction, labels)

        return loss, sample_specific_losses

    def update_target_network(self):
        weights = torch.nn.Module.state_dict(self.q_network)
        torch.nn.Module.load_state_dict(self.q_target_network, weights)

class ReplayBuffer():
    def __init__(self, min_p, batch_size_input, buffer_length_limit, alpha):
        self.min_p = min_p
        self.batch_size = batch_size_input
        self.buffer_length_limit = buffer_length_limit
        self.alpha = alpha

        self.buffer = collections.deque(maxlen=self.buffer_length_limit)
        self.idxs = None
        self.counter = 0
        self.w = np.zeros(self.buffer_length_limit)

    # Returns a sample batch of transitions extracted according to p (which depends on w)
    def sample_batch(self):
        self.w[:self.size()] = self.w[:self.size()] + self.min_p
        p = self.w**self.alpha/np.sum(self.w**self.alpha)
        self.idxs = np.random.choice(range(self.size()), self.batch_size, True, p[:self.size()])

        s = []
        a = []
        r = []
        ns = []

        for i in self.idxs:
            state, action, reward, new_state = self.buffer[i]
            s.append(state)
            a.append(action)
            r.append(reward)
            ns.append(new_state)

        s = np.array(s, dtype=np.float32)
        r = np.array(r, dtype=np.float32).reshape(-1, 1)
        a = np.array(a, dtype=np.int64).reshape(-1, 1)
        ns = np.array(ns, dtype=np.float32)

        return s, a, r, ns

    def add(self, transition):
        self.buffer.appendleft(transition)
        self.w[self.counter] = np.max(self.w) if self.has_more_than_enough_samples() else 1/self.batch_size
        self.counter = (self.counter % self.buffer_length_limit) + 1

    def update_weights(self, sample_specific_losses):
        self.w[self.idxs] = sample_specific_losses.detach().numpy().reshape(self.batch_size,)

    def has_enough_samples(self):
        return self.size() >= self.batch_size

    def has_more_than_enough_samples(self):
        return self.size() > self.batch_size

    def size(self):
        return len(self.buffer)

class Agent:
    # Initialise agent
    def __init__(self):
        self.episode_length = self.episode_length_was = 250

        self.episodes_so_far = 0
        self.n_steps_taken_since_beginning = 0
        self.episode_length_decay_rate = 25
        self.min_episode_length = 100

        self.current_state = None
        self.action = None

        self.test_greedy = False
        self.done = False

        self.gamma = 0.9
        self.mini_batch = 128
        self.lr = 0.001
        self.target_network_update_freq = 50

        self.eps = 1
        self.epsilon_decay_rate = 0.000025
        self.epsilon_decay_rate_prime = 0.0001

        self.buffer_size = 100000
        self.alpha_buffer = 0.7
        self.min_p = 0.000001

        self.dqn = DQN(self.lr)
        self.buffer = ReplayBuffer(self.min_p, self.mini_batch, self.buffer_size, self.alpha_buffer)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        episode_steps = self.n_steps_taken_since_beginning - self.episodes_so_far*self.episode_length

        if (episode_steps % self.episode_length==0):
            self.episodes_so_far += 1

            self.test_greedy = False
            self.episode_length = self.episode_length_was

            # Reduce length of episodes linearly by self.episode_length_decay_rate
            if (self.n_steps_taken_since_beginning >= 10000) and (self.episodes_so_far % 10 == 0):
                # Episode length is higher or equal 100 steps
                self.episode_length = max(self.min_episode_length, self.episode_length - self.episode_length_decay_rate)
                
                # Do greedy test
                self.episode_length_was = self.episode_length
                self.episode_length = 100
                self.test_greedy = True
            
            if not self.test_greedy and (self.episodes_so_far % 11 == 0):
                self.episode_length_was = self.episode_length
                self.episode_length = 500


            return True
        
        return False

    # Function to get the next action, epsilon greedy action selection
    def get_next_action(self, state):
        if not self.test_greedy and not self.done:
            self.current_state = state

            self.decay_epsilon()

            predictions = self.dqn.q_network.forward(torch.tensor(state))
            index_best_action = predictions.max(0)[1].item()

            p = np.full((3), self.eps/3)
            p[index_best_action] += 1 - self.eps
            self.action = np.random.choice(np.array([0, 1, 2]), p = p)

            self.n_steps_taken_since_beginning += 1

            continuous_action = self._discrete_action_to_continuous()

            return continuous_action

        else:
            self.n_steps_taken_since_beginning += 1
            self.current_state = state

            return self.get_greedy_action(state)

    # Train the q network with random sample batch
    def update(self):
        if self.buffer.has_enough_samples():
            batch = self.buffer.sample_batch()
            loss, sample_specific_losses = self.dqn.train_q_network(batch, self.gamma)

            return loss, sample_specific_losses
        
        return [], []

    # Decay epsilon depending on total number of steps done so far
    def decay_epsilon(self):
        relevant_steps = self.n_steps_taken_since_beginning-5000

        decay_rate = self.epsilon_decay_rate if self.n_steps_taken_since_beginning < 30000 else self.epsilon_decay_rate_prime

        # Not decaying to values smaller than 0.005
        if self.n_steps_taken_since_beginning > 5000:
            self.eps = max(0.005, 1 - decay_rate*(relevant_steps))

    # Function to convert discrete action to a continuous action, no left action
    def _discrete_action_to_continuous(self):
        if self.action == 0:
            action = np.array([0, 0.02], dtype=np.float32)
        elif self.action == 1:
            action = np.array([0.02, 0], dtype=np.float32)
        else:
            action = np.array([0, -0.02], dtype=np.float32)

        return action

    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if not self.done:
            if not self.test_greedy:
                # Close to goal
                if distance_to_goal < 0.05:
                    reward = 1 - distance_to_goal
                # Hit wall
                elif self.current_state is next_state:
                    reward = -0.1*distance_to_goal
                # Effectively goes right
                elif self.current_state[0] < next_state[0]:
                    reward = (1 - distance_to_goal)
                else:
                    reward = 0

                self.buffer.add((self.current_state, self.action, reward, next_state))

                self.current_state = next_state

                loss, sample_specific_losses = self.update()

                relevant_steps_so_far = self.n_steps_taken_since_beginning - self.mini_batch

                if relevant_steps_so_far >= 0:
                    self.buffer.update_weights(sample_specific_losses)

                    if (relevant_steps_so_far != 0) and ((relevant_steps_so_far % self.target_network_update_freq) == 0):
                        self.dqn.update_target_network()

            else:
                self.current_state = next_state

                # Stop training
                if distance_to_goal < 0.05:
                    self.done = True

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        predictions = self.dqn.q_network.forward(torch.tensor(state))
        self.action = predictions.max(0)[1].item()
        action = self._discrete_action_to_continuous()

        return action