import random
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Define all possible categories
ALL_CATEGORIES = ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 
                 'four_of_a_kind', 'full_house', 'straight', 'yams', 
                 'plus', 'minus']

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

def play_turn(game, strategy_function):
    """Play a single turn using the provided strategy function"""
    # First roll - roll all 5 dice
    game.roll_dice([0, 1, 2, 3, 4])
    
    # Strategy function decides which dice to keep and which category to target
    for roll_num in range(2):  # We already did the first roll, so 2 more possible
        keep_indices = strategy_function(game.dice.copy(), game.get_available_categories(), roll_num + 1)
        
        # If all dice are kept, break out of the rolling phase
        if len(keep_indices) == 5:
            break
        
        # Roll the non-kept dice
        roll_indices = [i for i in range(5) if i not in keep_indices]
        game.roll_dice(roll_indices)
    
    # After rolling phase, decide which category to assign
    category = strategy_function(game.dice.copy(), game.get_available_categories(), 3)
    
    # Assign the score to the chosen category
    game.assign_score(category)
    
    return game

def play_game(strategy_function):
    """Play a full game using the provided strategy function"""
    game = YamsGame()
    
    while not game.is_game_over():
        game = play_turn(game, strategy_function)
    
    return game.get_total_score(), game.scorecard

# Strategy 1: Random
def random_strategy(dice, available_categories, roll_num):
    """A simple random strategy for testing"""
    if roll_num < 3:  # Decide which dice to keep
        # Randomly keep some dice
        keep_prob = 0.5  # 50% chance to keep each die
        keep_indices = [i for i in range(5) if random.random() < keep_prob]
        return keep_indices
    else:  # Decide which category to score
        return random.choice(available_categories)

# Strategy 2: Heuristic
def heuristic_strategy(dice, available_categories, roll_num):
    """A heuristic-based strategy"""
    # Create a temporary game to calculate scores
    temp_game = YamsGame()
    temp_game.dice = dice.copy()
    
    if roll_num < 3:  # Decide which dice to keep
        # Count the occurrences of each value
        counts = Counter(dice)
        
        # Keep dice that appear multiple times (potential for matching categories)
        keep_indices = []
        for i, value in enumerate(dice):
            # Keep dice that are part of pairs or better
            if counts[value] >= 2:
                keep_indices.append(i)
            # Keep high values if targeting plus
            elif value >= 4 and 'plus' in available_categories:
                keep_indices.append(i)
            # Keep low values if targeting minus
            elif value <= 3 and 'minus' in available_categories:
                keep_indices.append(i)
        
        return keep_indices
    
    else:  # Decide which category to score
        # Calculate the score for each available category
        scores = [(category, temp_game.calculate_score(category)) for category in available_categories]
        
        # Choose the category with the highest score
        return max(scores, key=lambda x: x[1])[0]

# Strategy 3: Matching Focus
def matching_focus_strategy(dice, available_categories, roll_num):
    """A strategy that focuses on getting matching dice (pairs, three of a kind, etc.)"""
    # Create a temporary game to calculate scores
    temp_game = YamsGame()
    temp_game.dice = dice.copy()
    
    if roll_num < 3:  # Decide which dice to keep
        # Count the occurrences of each value
        counts = Counter(dice)
        
        # Keep dice that have multiple occurrences
        keep_indices = []
        for i, value in enumerate(dice):
            if counts[value] >= 2:
                keep_indices.append(i)
        
        return keep_indices
    
    else:  # Decide which category to score
        # Prioritize categories that involve matching dice
        priority_categories = ['yams', 'four_of_a_kind', 'full_house']
        
        for category in priority_categories:
            if category in available_categories and temp_game.calculate_score(category) > 0:
                return category
        
        # If no matching category is available or has a score, choose the highest-scoring category
        scores = [(category, temp_game.calculate_score(category)) for category in available_categories]
        return max(scores, key=lambda x: x[1])[0]

# Strategy 4: Straight Focus
def straight_focus_strategy(dice, available_categories, roll_num):
    """A strategy that focuses on getting straights"""
    # Create a temporary game to calculate scores
    temp_game = YamsGame()
    temp_game.dice = dice.copy()
    
    if roll_num < 3:  # Decide which dice to keep
        # Get sorted unique dice values
        sorted_unique_dice = sorted(set(dice))
        
        # Find potential consecutive sequences
        consecutive_sequences = []
        current_sequence = [sorted_unique_dice[0]]
        
        for i in range(1, len(sorted_unique_dice)):
            if sorted_unique_dice[i] == sorted_unique_dice[i-1] + 1:
                current_sequence.append(sorted_unique_dice[i])
            else:
                consecutive_sequences.append(current_sequence)
                current_sequence = [sorted_unique_dice[i]]
        
        consecutive_sequences.append(current_sequence)
        
        # Find the longest consecutive sequence
        longest_sequence = max(consecutive_sequences, key=len)
        
        # Keep dice that are part of the longest sequence
        keep_indices = [i for i, value in enumerate(dice) if value in longest_sequence]
        
        # If we have less than 3 consecutive values, see if we can create a better sequence
        if len(longest_sequence) < 3:
            # Try to keep dice that are close to each other in value
            for start in range(1, 4):  # Start can be 1, 2, or 3
                potential_straight = list(range(start, start + 4))
                matching = [i for i, value in enumerate(dice) if value in potential_straight]
                if len(matching) > len(keep_indices):
                    keep_indices = matching
        
        return keep_indices
    
    else:  # Decide which category to score
        # Prioritize the straight category if available and has a score
        if 'straight' in available_categories and temp_game.calculate_score('straight') > 0:
            return 'straight'
        
        # If straight is not available or has no score, choose the highest-scoring category
        scores = [(category, temp_game.calculate_score(category)) for category in available_categories]
        return max(scores, key=lambda x: x[1])[0]

def evaluate_strategy(strategy_function, num_games=1000):
    """Evaluate a strategy by playing many games and averaging the scores"""
    scores = []
    
    for _ in range(num_games):
        final_score, _ = play_game(strategy_function)
        scores.append(final_score)
    
    return np.mean(scores), np.std(scores)

def evaluate_strategies(num_games=1000):
    """Evaluate different strategies and compare their performance"""
    strategies = {
        'Random': random_strategy,
        'Heuristic': heuristic_strategy,
        'Matching Focus': matching_focus_strategy,
        'Straight Focus': straight_focus_strategy
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        mean_score, std_score = evaluate_strategy(strategy, num_games=num_games)
        results[name] = (mean_score, std_score)
        print(f"{name} Strategy: Mean Score = {mean_score:.2f}, Std Dev = {std_score:.2f}")
    
    return results

def train_ml_models(num_games=5000):
    """Train machine learning models to predict the best actions"""
    # Collect data from games played by the best strategies
    strategies = [heuristic_strategy, matching_focus_strategy, straight_focus_strategy]
    
    # Collect training data
    training_data = []
    
    for strategy in strategies:
        for _ in range(num_games // len(strategies)):
            game = YamsGame()
            
            while not game.is_game_over():
                # First roll - roll all 5 dice
                game.roll_dice([0, 1, 2, 3, 4])
                
                for roll_num in range(1, 3):  # Rolls 1 and 2
                    # Record the state and action
                    state = game.dice.copy()
                    available_cats = game.get_available_categories()
                    
                    # Get the action from the strategy
                    action = strategy(state, available_cats, roll_num)
                    
                    # Record the data
                    training_data.append({
                        'dice': state,
                        'available_categories': available_cats,
                        'roll_num': roll_num,
                        'action': action,
                        'strategy': strategy.__name__
                    })
                    
                    # Roll the non-kept dice
                    roll_indices = [i for i in range(5) if i not in action]
                    game.roll_dice(roll_indices)
                
                # After the final roll, record the state and action
                state = game.dice.copy()
                available_cats = game.get_available_categories()
                action = strategy(state, available_cats, 3)
                
                training_data.append({
                    'dice': state,
                    'available_categories': available_cats,
                    'roll_num': 3,
                    'action': action,
                    'strategy': strategy.__name__
                })
                
                # Assign the score
                game.assign_score(action)
    
    # Prepare features and targets for the models
    X_keep = []
    y_keep = []
    X_category = []
    y_category = []
    
    for data_point in training_data:
        # Create feature vector
        dice = data_point['dice']
        available_cats = data_point['available_categories']
        roll_num = data_point['roll_num']
        
        # Create feature vector: dice values + available categories + roll_num
        feature = dice.copy()
        feature.extend([1 if cat in available_cats else 0 for cat in ALL_CATEGORIES])
        feature.append(roll_num)
        
        if roll_num < 3:  # Data for keep/reroll decisions
            X_keep.append(feature)
            
            # Convert action (indices to keep) to binary vector
            keep_vector = [1 if i in data_point['action'] else 0 for i in range(5)]
            y_keep.append(keep_vector)
        else:  # Data for category selection
            X_category.append(feature)
            
            # Convert action (category) to index
            category_index = ALL_CATEGORIES.index(data_point['action'])
            y_category.append(category_index)
    
    # Train models for dice-keeping (rolls 1-2)
    keep_models = []
    for i in range(5):  # One model for each die
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_keep, [y[i] for y in y_keep])
        keep_models.append(model)
    
    # Train model for category selection (roll 3)
    category_model = RandomForestRegressor(n_estimators=100)
    category_model.fit(X_category, y_category)
    
    return keep_models, category_model

def ml_strategy(dice, available_categories, roll_num, keep_models, category_model):
    """A strategy that uses the trained machine learning models"""
    # Create feature vector
    feature = dice.copy()
    feature.extend([1 if cat in available_categories else 0 for cat in ALL_CATEGORIES])
    feature.append(roll_num)
    
    if roll_num < 3:  # Decide which dice to keep
        # Predict whether to keep each die
        keep_probs = [model.predict([feature])[0] for model in keep_models]
        
        # Keep dice with probability > 0.5
        keep_indices = [i for i, prob in enumerate(keep_probs) if prob > 0.5]
        
        return keep_indices
    
    else:  # Decide which category to score
        # Predict the best category index
        category_index = int(category_model.predict([feature])[0] + 0.5)  # Round to nearest integer
        
        # Make sure the category is available
        while ALL_CATEGORIES[category_index] not in available_categories:
            # If not available, find the next best category
            category_index = (category_index + 1) % len(ALL_CATEGORIES)
        
        return ALL_CATEGORIES[category_index]

# Main function to run the code
def main():
    # First, evaluate different predefined strategies
    print("Evaluating predefined strategies...")
    results = evaluate_strategies(num_games=1000)
    
    # Train the machine learning models
    print("\nTraining machine learning models...")
    keep_models, category_model = train_ml_models(num_games=5000)
    
    # Create the ML strategy function
    ml_strategy_func = lambda dice, available_categories, roll_num: ml_strategy(
        dice, available_categories, roll_num, keep_models, category_model)
    
    # Evaluate the ML strategy
    print("\nEvaluating machine learning strategy...")
    ml_mean, ml_std = evaluate_strategy(ml_strategy_func, num_games=1000)
    
    print(f"ML Strategy: Mean Score = {ml_mean:.2f}, Std Dev = {ml_std:.2f}")
    
    # Compare all strategies
    print("\nStrategy Comparison:")
    for name, (mean, std) in results.items():
        print(f"{name} Strategy: Mean Score = {mean:.2f}, Std Dev = {std:.2f}")
    print(f"ML Strategy: Mean Score = {ml_mean:.2f}, Std Dev = {ml_std:.2f}")
    
    # Play a sample game with the ML strategy to see its decisions
    print("\nPlaying a sample game with ML strategy...")
    game = YamsGame()
    
    turn = 1
    while not game.is_game_over():
        print(f"\nTurn {turn}:")
        
        # First roll - roll all 5 dice
        dice = game.roll_dice([0, 1, 2, 3, 4])
        print(f"Initial roll: {dice}")
        
        for roll_num in range(1, 3):
            # Get the action from the ML strategy
            keep_indices = ml_strategy(dice, game.get_available_categories(), roll_num, keep_models, category_model)
            kept_dice = [dice[i] for i in keep_indices]
            print(f"Roll {roll_num}: Keeping dice at positions {keep_indices} with values {kept_dice}")
            
            # If all dice are kept, break out of the rolling phase
            if len(keep_indices) == 5:
                break
            
            # Roll the non-kept dice
            roll_indices = [i for i in range(5) if i not in keep_indices]
            dice = game.roll_dice(roll_indices)
            print(f"New dice after roll {roll_num+1}: {dice}")
        
        # After rolling phase, decide which category to assign
        category = ml_strategy(dice, game.get_available_categories(), 3, keep_models, category_model)
        score = game.calculate_score(category)
        
        # Assign the score to the chosen category
        game.assign_score(category)
        
        print(f"Assigned to category '{category}' for {score} points")
        print(f"Current scorecard: {game.scorecard}")
        
        turn += 1
    
    print("\nFinal scorecard:")
    for category, score in game.scorecard.items():
        print(f"{category}: {score}")
    
    # Calculate upper section score for bonus calculation
    upper_section_score = sum(game.scorecard[cat] for cat in 
                            ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes'])
    bonus = 35 if upper_section_score >= 63 else 0
    
    print(f"\nUpper section score: {upper_section_score}")
    print(f"Bonus: {bonus}")
    print(f"Total score: {game.get_total_score()}")

# Analysis of optimization potential
def analyze_decisions(num_games=1000):
    """Analyze how often different categories are selected and their average scores"""
    # Play games with each strategy and record category selections
    strategies = {
        'Random': random_strategy,
        'Heuristic': heuristic_strategy,
        'Matching Focus': matching_focus_strategy,
        'Straight Focus': straight_focus_strategy
    }
    
    category_selections = {name: {cat: 0 for cat in ALL_CATEGORIES} for name in strategies}
    category_scores = {name: {cat: [] for cat in ALL_CATEGORIES} for name in strategies}
    
    for strategy_name, strategy_func in strategies.items():
        for _ in range(num_games):
            game = YamsGame()
            
            while not game.is_game_over():
                # First roll
                game.roll_dice([0, 1, 2, 3, 4])
                
                # Strategy handles the next two rolls
                for roll_num in range(1, 3):
                    keep_indices = strategy_func(game.dice.copy(), game.get_available_categories(), roll_num)
                    
                    # If all dice are kept, break
                    if len(keep_indices) == 5:
                        break
                    
                    # Roll non-kept dice
                    roll_indices = [i for i in range(5) if i not in keep_indices]
                    game.roll_dice(roll_indices)
                
                # Choose category
                category = strategy_func(game.dice.copy(), game.get_available_categories(), 3)
                score = game.calculate_score(category)
                
                # Record category selection and score
                category_selections[strategy_name][category] += 1
                category_scores[strategy_name][category].append(score)
                
                # Assign score
                game.assign_score(category)
    
    # Calculate average scores per category for each strategy
    avg_scores = {name: {} for name in strategies}
    
    for strategy_name in strategies:
        for category in ALL_CATEGORIES:
            scores = category_scores[strategy_name][category]
            if scores:
                avg_scores[strategy_name][category] = sum(scores) / len(scores)
            else:
                avg_scores[strategy_name][category] = 0
    
    # Print analysis
    print("\nStrategy Analysis:")
    for strategy_name in strategies:
        print(f"\n{strategy_name} Strategy:")
        print("Category\t| Times Selected\t| Average Score")
        print("-" * 50)
        
        for category in ALL_CATEGORIES:
            selections = category_selections[strategy_name][category]
            avg_score = avg_scores[strategy_name][category]
            
            print(f"{category}\t| {selections}\t\t| {avg_score:.2f}")
    
    return category_selections, avg_scores

if __name__ == "__main__":
    main()
    
    # Run the analysis
    print("\nRunning strategy analysis...")
    analyze_decisions(num_games=500)