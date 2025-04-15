import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, Counter

# Import YamsGame from your existing implementation
# For this example I'm assuming it's already defined with the same interface

class DQNetwork(nn.Module):
    """Neural network to approximate Q-function for Yams"""
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PyTorchDQNAgent:
    """PyTorch implementation of DQN agent for Yams"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None):
        """Determine action based on state"""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Convert numpy array to PyTorch tensor and move to GPU
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action values
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            act_values = self.model(state_tensor)
        self.model.train()  # Set back to training mode
        
        # Filter for valid actions only
        act_values_np = act_values.cpu().numpy()[0]
        valid_values = [(i, act_values_np[i]) for i in valid_actions]
        return max(valid_values, key=lambda x: x[1])[0]
    
    def replay(self, batch_size):
        """Train the model with experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Process batch more efficiently
        states = np.array([m[0][0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3][0] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Predict Q-values for current states
        self.model.train()
        pred = self.model(states_tensor)
        
        # Select Q-values for chosen actions
        pred = pred.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Calculate target Q-values
        with torch.no_grad():
            next_pred = self.model(next_states_tensor)
            max_next_pred = torch.max(next_pred, 1)[0]
            
        # Calculate target using Bellman equation
        target = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_pred
        
        # Compute loss and update model
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, file_path):
        """Save model weights"""
        torch.save(self.model.state_dict(), file_path)
        
    def load(self, file_path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

def encode_state(dice, available_categories, roll_num):
    """Encode the game state (from your original implementation)"""
    # This should be the same function from your tensorflow implementation
    # Keeping as a placeholder - insert your actual implementation
    # Example:
    dice_encoding = np.zeros(30)  # 5 dice x 6 values
    for i, value in enumerate(dice):
        if value > 0:
            dice_encoding[i*6 + (value-1)] = 1
            
    categories_encoding = np.zeros(len(ALL_CATEGORIES))
    for cat in available_categories:
        categories_encoding[ALL_CATEGORIES.index(cat)] = 1
        
    roll_encoding = np.zeros(3)
    roll_encoding[roll_num-1] = 1
    
    return np.concatenate([dice_encoding, categories_encoding, roll_encoding])

def train_keep_agent_pytorch(episodes=1000):
    """Train an agent to decide which dice to keep using PyTorch"""
    state_size = 30 + len(ALL_CATEGORIES) + 3  # dice + categories + roll number
    action_size = 2**5  # all possible keep/reroll combinations for 5 dice
    
    agent = PyTorchDQNAgent(state_size, action_size)
    batch_size = 32
    
    scores = []
    
    for e in range(episodes):
        game = YamsGame()
        
        while not game.is_game_over():
            # First roll - roll all 5 dice
            game.roll_dice([0, 1, 2, 3, 4])
            
            for roll_num in range(1, 3):  # Rolls 1 and 2
                # Current state
                state = encode_state(game.dice, game.get_available_categories(), roll_num)
                state = np.reshape(state, [1, state_size])
                
                # Get action (which dice to keep)
                binary_actions = [bin(i)[2:].zfill(5) for i in range(action_size)]
                valid_actions = [i for i, binary in enumerate(binary_actions) if all(b == '1' for b in binary) == False]
                
                action = agent.act(state, valid_actions)
                
                # Convert action index to binary representation
                binary = bin(action)[2:].zfill(5)
                keep_indices = [i for i, bit in enumerate(binary) if bit == '1']
                
                # Roll the non-kept dice
                roll_indices = [i for i, bit in enumerate(binary) if bit == '0']
                old_dice = game.dice.copy()
                game.roll_dice(roll_indices)
                
                # Next state
                next_state = encode_state(game.dice, game.get_available_categories(), roll_num + 1 if roll_num < 2 else 3)
                next_state = np.reshape(next_state, [1, state_size])
                
                # Reward: improvement in potential score
                old_potential = max(game.calculate_score(cat) for cat in game.get_available_categories())
                new_potential = max(game.calculate_score(cat) for cat in game.get_available_categories())
                reward = new_potential - old_potential
                
                # Store experience
                done = roll_num == 2 or all(bit == '1' for bit in binary)
                agent.memorize(state, action, reward, next_state, done)
                
                # If all dice kept or last roll, break
                if done:
                    break
            
            # After rolling phase, choose category with highest score
            scores_per_category = [(cat, game.calculate_score(cat)) for cat in game.get_available_categories()]
            best_category = max(scores_per_category, key=lambda x: x[1])[0]
            
            # Assign score
            game.assign_score(best_category)
        
        # After game ends, train with replay
        agent.replay(min(batch_size, len(agent.memory)))
        
        # Log progress
        total_score = game.get_total_score()
        scores.append(total_score)
        
        if e % 100 == 0:
            print(f"Episode: {e}/{episodes}, Score: {total_score}, Epsilon: {agent.epsilon:.2f}")
    
    # Save the model
    agent.save("yams_keep_agent_pytorch.pth")
    
    return agent, scores

# Run with GPU monitoring
if __name__ == "__main__":
    print("Starting PyTorch DQN training for Yams...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    agent, scores = train_keep_agent_pytorch(episodes=1000)
    
    # Plot scores if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(scores)
        plt.title('PyTorch DQN Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        plt.savefig('pytorch_training_scores.png')
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting. Scores saved in memory.")