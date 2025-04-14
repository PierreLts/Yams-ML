import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque, Counter
import pickle
import matplotlib.pyplot as plt

# Import the base game from our first implementation
# In a real implementation, you would import YamsGame from the first file
class YamsGame:
    def __init__(self):
        # Initialize the scorecard
        self.scorecard = {category: None for category in ALL_CATEGORIES}
        self.dice = [0, 0, 0, 0, 0]
        self.turns_played = 0
        
    def roll_dice(self, dice_to_roll):
        """Roll the specified dice indices (0-4)"""
        for i in dice_to_roll:
            self.dice[i] = random.randint(1, 6)
        return self.dice
    
    def calculate_score(self, category):
        """Calculate the score for a given category based on current dice"""
        dice_counts = Counter(self.dice)
        
        if category in ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes']:
            # Upper section scoring
            number = {'ones': 1, 'twos': 2, 'threes': 3, 'fours': 4, 'fives': 5, 'sixes': 6}[category]
            return number * dice_counts[number]
        
        elif category == 'four_of_a_kind':
            # Check if there are at least 4 of any number
            if any(count >= 4 for count in dice_counts.values()):
                return 24
            return 0
        
        elif category == 'full_house':
            # Check for 3 of one number and 2 of another
            if sorted(dice_counts.values()) == [2, 3] and len(dice_counts) == 2:
                return 25
            return 0
        
        elif category == 'straight':
            # Sort the unique dice values
            sorted_unique_dice = sorted(set(self.dice))
            
            # Check for straights
            if len(sorted_unique_dice) >= 5:
                # Check if values form a consecutive sequence
                consecutive_count = 1
                max_consecutive = 1
                
                for i in range(1, len(sorted_unique_dice)):
                    if sorted_unique_dice[i] == sorted_unique_dice[i-1] + 1:
                        consecutive_count += 1
                        max_consecutive = max(max_consecutive, consecutive_count)
                    else:
                        consecutive_count = 1
                
                if max_consecutive >= 5:
                    return 40  # 5-length straight
                elif max_consecutive >= 4:
                    return 30  # 4-length straight
            
            # Check specifically for 4-length straight if we don't have 5 unique dice
            elif len(sorted_unique_dice) >= 4:
                consecutive_count = 1
                max_consecutive = 1
                
                for i in range(1, len(sorted_unique_dice)):
                    if sorted_unique_dice[i] == sorted_unique_dice[i-1] + 1:
                        consecutive_count += 1
                        max_consecutive = max(max_consecutive, consecutive_count)
                    else:
                        consecutive_count = 1
                
                if max_consecutive >= 4:
                    return 30  # 4-length straight
            
            return 0
        
        elif category == 'yams':
            # Check for 5 of the same number
            if any(count == 5 for count in dice_counts.values()):
                return 50
            return 0
        
        elif category == 'plus':
            # Sum of all dice
            return sum(self.dice)
        
        elif category == 'minus':
            # Subtract all dice values (negative score)
            return -sum(self.dice)
        
        return 0
    
    def assign_score(self, category):
        """Assign the current dice score to a category"""
        if self.scorecard[category] is not None:
            raise ValueError(f"Category {category} already scored")
        
        self.scorecard[category] = self.calculate_score(category)
        self.turns_played += 1
        
    def get_total_score(self):
        """Calculate total score including upper section bonus"""
        # Calculate upper section score
        upper_section_score = sum(self.scorecard[cat] or 0 for cat in 
                                ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes'])
        
        # Apply bonus if upper section score is at least 63
        bonus = 35 if upper_section_score >= 63 else 0
        
        # Calculate total score
        total_score = sum(score or 0 for score in self.scorecard.values() if score is not None) + bonus
        
        return total_score
    
    def is_game_over(self):
        """Check if the game is over (all categories filled)"""
        return self.turns_played >= 12
    
    def get_available_categories(self):
        """Return a list of categories that haven't been scored yet"""
        return [category for category, score in self.scorecard.items() if score is None]

# Define all possible categories
ALL_CATEGORIES = ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 
                 'four_of_a_kind', 'full_house', 'straight', 'yams', 
                 'plus', 'minus']

class DQNAgent:
    """Deep Q-Network agent for learning Yams strategies"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        """Neural network to approximate Q-function"""
        model = keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None):
        """Determine action based on state"""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        act_values = self.model.predict(state, verbose=0)
        
        # Filter for valid actions only
        valid_values = [(i, act_values[0][i]) for i in valid_actions]
        return max(valid_values, key=lambda x: x[1])[0]
    
    def replay(self, batch_size):
        """Train the model with experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            targets[i] = target_f[0]
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def encode_dice(dice):
    """Encode dice as a one-hot vector"""
    encoding = np.zeros(30)  # 5 dice x 6 possible values
    
    for i, value in enumerate(dice):
        if value > 0:  # Skip dice that aren't rolled yet
            encoding[i*6 + (value-1)] = 1
    
    return encoding

def encode_available_categories(available_categories):
    """Encode available categories as a binary vector"""
    encoding = np.zeros(len(ALL_CATEGORIES))
    
    for cat in available_categories:
        encoding[ALL_CATEGORIES.index(cat)] = 1
    
    return encoding

def encode_state(dice, available_categories, roll_num):
    """Encode the full game state"""
    dice_encoding = encode_dice(dice)
    categories_encoding = encode_available_categories(available_categories)
    
    # One-hot encode the roll number
    roll_encoding = np.zeros(3)
    roll_encoding[roll_num-1] = 1
    
    return np.concatenate([dice_encoding, categories_encoding, roll_encoding])

def train_keep_agent(episodes=1000):
    """Train an agent to decide which dice to keep after each roll"""
    state_size = 30 + len(ALL_CATEGORIES) + 3  # dice + categories + roll number
    action_size = 2**5  # all possible keep/reroll combinations for 5 dice
    
    agent = DQNAgent(state_size, action_size)
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
    
    return agent, scores

def train_category_agent(episodes=1000):
    """Train an agent to decide which category to select after the final roll"""
    state_size = 30 + len(ALL_CATEGORIES) + 3  # dice + categories + roll number
    action_size = len(ALL_CATEGORIES)
    
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    
    scores = []
    
    for e in range(episodes):
        game = YamsGame()
        
        while not game.is_game_over():
            # First roll - roll all 5 dice
            game.roll_dice([0, 1, 2, 3, 4])
            
            # Use a simple heuristic for the rolling phase
            for roll_num in range(1, 3):
                # Count the occurrences of each value
                counts = Counter(game.dice)
                
                # Keep dice that have multiple occurrences
                keep_indices = []
                for i, value in enumerate(game.dice):
                    if counts[value] >= 2:
                        keep_indices.append(i)
                
                # If all dice kept, break
                if len(keep_indices) == 5:
                    break
                
                # Roll the non-kept dice
                roll_indices = [i for i in range(5) if i not in keep_indices]
                game.roll_dice(roll_indices)
            
            # Final state - decide which category to choose
            state = encode_state(game.dice, game.get_available_categories(), 3)
            state = np.reshape(state, [1, state_size])
            
            # Get valid actions (available categories)
            valid_actions = [ALL_CATEGORIES.index(cat) for cat in game.get_available_categories()]
            
            # Choose category
            action = agent.act(state, valid_actions)
            category = ALL_CATEGORIES[action]
            
            # Reward is the score for the category
            score = game.calculate_score(category)
            reward = score
            
            # Next state (for simplicity, use a dummy state)
            next_state = np.zeros_like(state)
            
            # Store experience
            done = True  # Category selection is always a terminal state
            agent.memorize(state, action, reward, next_state, done)
            
            # Assign score
            game.assign_score(category)
        
        # After game ends, train with replay
        agent.replay(min(batch_size, len(agent.memory)))
        
        # Log progress
        total_score = game.get_total_score()
        scores.append(total_score)
        
        if e % 100 == 0:
            print(f"Episode: {e}/{episodes}, Score: {total_score}, Epsilon: {agent.epsilon:.2f}")
    
    return agent, scores

def combined_dqn_strategy(dice, available_categories, roll_num, keep_agent, category_agent):
    """Strategy that uses both DQN agents for decision making"""
    # Encode the state
    state_size = 30 + len(ALL_CATEGORIES) + 3
    state = encode_state(dice, available_categories, roll_num)
    state = np.reshape(state, [1, state_size])
    
    if roll_num < 3:  # Decide which dice to keep
        # Get valid actions
        action_size = 2**5
        binary_actions = [bin(i)[2:].zfill(5) for i in range(action_size)]
        valid_actions = [i for i, binary in enumerate(binary_actions) if all(b == '1' for b in binary) == False]
        
        # Choose action
        action = keep_agent.act(state, valid_actions)
        
        # Convert to keep indices
        binary = bin(action)[2:].zfill(5)
        keep_indices = [i for i, bit in enumerate(binary) if bit == '1']
        
        return keep_indices
    
    else:  # Decide which category to choose
        # Get valid actions
        valid_actions = [ALL_CATEGORIES.index(cat) for cat in available_categories]
        
        # Choose action
        action = category_agent.act(state, valid_actions)
        
        # Convert to category
        category = ALL_CATEGORIES[action]
        
        return category

def evaluate_dqn_strategy(keep_agent, category_agent, num_games=1000):
    """Evaluate the DQN strategy by playing many games"""
    scores = []
    
    for _ in range(num_games):
        game = YamsGame()
        
        while not game.is_game_over():
            # First roll
            game.roll_dice([0, 1, 2, 3, 4])
            
            # Use the DQN strategy for decision making
            for roll_num in range(1, 3):
                keep_indices = combined_dqn_strategy(
                    game.dice, game.get_available_categories(), roll_num, keep_agent, category_agent)
                
                # If all dice kept, break
                if len(keep_indices) == 5:
                    break
                
                # Roll non-kept dice
                roll_indices = [i for i in range(5) if i not in keep_indices]
                game.roll_dice(roll_indices)
            
            # Choose category
            category = combined_dqn_strategy(
                game.dice, game.get_available_categories(), 3, keep_agent, category_agent)
            
            # Assign score
            game.assign_score(category)
        
        # Record final score
        scores.append(game.get_total_score())
    
    return np.mean(scores), np.std(scores)

def plot_learning_progress(scores, title="Learning Progress"):
    """Plot the learning progress (scores over episodes)"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    
    # Add moving average
    window_size = 100
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), moving_avg, 'r-', linewidth=2)
    
    plt.grid(True)
    plt.show()

def play_dqn_sample_game(keep_agent, category_agent):
    """Play a sample game with the DQN strategy to see decisions"""
    game = YamsGame()
    turn = 1
    
    while not game.is_game_over():
        print(f"\nTurn {turn}:")
        
        # First roll
        dice = game.roll_dice([0, 1, 2, 3, 4])
        print(f"Initial roll: {dice}")
        
        for roll_num in range(1, 3):
            # Get action
            keep_indices = combined_dqn_strategy(
                dice, game.get_available_categories(), roll_num, keep_agent, category_agent)
            
            kept_dice = [dice[i] for i in keep_indices]
            print(f"Roll {roll_num}: Keeping dice at positions {keep_indices} with values {kept_dice}")
            
            # If all dice kept, break
            if len(keep_indices) == 5:
                break
            
            # Roll non-kept dice
            roll_indices = [i for i in range(5) if i not in keep_indices]
            dice = game.roll_dice(roll_indices)
            print(f"New dice after roll {roll_num+1}: {dice}")
        
        # Choose category
        category = combined_dqn_strategy(
            dice, game.get_available_categories(), 3, keep_agent, category_agent)
        
        score = game.calculate_score(category)
        
        # Assign score
        game.assign_score(category)
        
        print(f"Assigned to category '{category}' for {score} points")
        print(f"Current scorecard: {game.scorecard}")
        
        turn += 1
    
    print("\nFinal scorecard:")
    for category, score in game.scorecard.items():
        print(f"{category}: {score}")
    
    # Calculate upper section score for bonus
    upper_section_score = sum(game.scorecard[cat] for cat in 
                           ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes'])
    bonus = 35 if upper_section_score >= 63 else 0
    
    print(f"\nUpper section score: {upper_section_score}")
    print(f"Bonus: {bonus}")
    print(f"Total score: {game.get_total_score()}")

def main():
    # Train the DQN agents
    print("Training keep decision agent...")
    keep_agent, keep_scores = train_keep_agent(episodes=500)
    
    print("\nTraining category selection agent...")
    category_agent, category_scores = train_category_agent(episodes=500)
    
    # Save the trained agents
    keep_agent.model.save("yams_keep_agent.h5")
    category_agent.model.save("yams_category_agent.h5")
    
    # Plot learning progress
    plot_learning_progress(keep_scores, "Keep Agent Learning Progress")
    plot_learning_progress(category_scores, "Category Agent Learning Progress")
    
    # Evaluate the trained agents
    print("\nEvaluating DQN strategy...")
    mean_score, std_score = evaluate_dqn_strategy(keep_agent, category_agent, num_games=1000)
    print(f"DQN Strategy: Mean Score = {mean_score:.2f}, Std Dev = {std_score:.2f}")
    
    # Play a sample game
    print("\nPlaying a sample game with DQN strategy...")
    play_dqn_sample_game(keep_agent, category_agent)

if __name__ == "__main__":
    main()