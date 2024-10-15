import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Convention: every class that implements a neural network derives nn.Module.
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        # Call super constructor right away.
        super(DeepQNetwork, self).__init__()
        # Set learning rate
        self.lr = lr

        # Input dims
        self.input_dims = input_dims

        # FC1 dims
        self.fc1_dims = fc1_dims

        # FC2 dims
        self.fc2_dims = fc2_dims

        self.n_actions = n_actions

        # Three layers to a neural network - simple network
        # First, it converts inputs (of any number of dims) in the first layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # Adds Hidden Layers
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Output of Deep Neural Network - get action.
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # Use optimizer (Adam)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Calculate loss
        self.loss = nn.MSELoss()
        # Handle CUDA if available
        self.device = T.device('cuda:0' if T.cuda.is_available() else "cpu")
        # Send network to device.
        self.to(self.device)

    # Handles forward propagation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Get raw numbers for actions.
        actions = self.fc3(x)

        return actions


class Agent:
    # Gamma = discount factor
    # Epsilon = explore/exploit (0 <= epsilon <= 1)
    # Epsilon_min = minimum value of epsilon
    # Eps_dis = multiplicative (linear dependence, inverse sqrt, whatever, as long as it's a decreasing function)
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_min=0.01,
                 eps_dis=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dis = eps_dis
        self.lr = lr
        self.action_space = list(range(n_actions))
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        # State memory
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        # New states for TD learning
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        # Action memory
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        # Terminal states: the game is done, future value is 0.
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Take best known action: take observation,
            # turn to PyTorch tensor and send to device.
            # B/c of the way the deep learning network is
            # set up, we need the observation to be in the brackets.
            state = T.tensor(np.array(observation)).to(self.Q_eval.device)
            # Use forward propagation to evaluate q-values of all actions.
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            # Pick a random action.
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # Start learning as soon as it "learns" a batch size of memory
        if self.mem_cntr < self.batch_size:
            return
        # Zero the gradient on optimizer.
        self.Q_eval.optimizer.zero_grad()

        # Calculate max memory
        max_mem = min(self.mem_cntr, self.mem_size)

        # Don't select memories more than once.
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # You need it to perform proper array slicing.
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # Use Bellman equation.
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        new_epsilon = self.epsilon * (1 - self.eps_dis)
        self.epsilon = new_epsilon if new_epsilon > self.eps_min else self.eps_min
