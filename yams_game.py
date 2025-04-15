# Replace the scikit-learn RandomForest import with cuML version
# Replace this line:
# from sklearn.ensemble import RandomForestRegressor

# With these imports:
import cudf
import numpy as np
from cuml.ensemble import RandomForestRegressor

# Then update the train_ml_models function:
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
    
    # Convert to GPU DataFrames
    X_keep_gpu = cudf.DataFrame(np.array(X_keep))
    X_category_gpu = cudf.DataFrame(np.array(X_category))
    
    # Train models for dice-keeping (rolls 1-2)
    keep_models = []
    for i in range(5):  # One model for each die
        y_keep_i = cudf.Series(np.array([y[i] for y in y_keep]))
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_keep_gpu, y_keep_i)
        keep_models.append(model)
    
    # Train model for category selection (roll 3)
    y_category_gpu = cudf.Series(np.array(y_category))
    category_model = RandomForestRegressor(n_estimators=100)
    category_model.fit(X_category_gpu, y_category_gpu)
    
    return keep_models, category_model

# Update the ml_strategy function to handle cuML models
def ml_strategy(dice, available_categories, roll_num, keep_models, category_model):
    """A strategy that uses the trained machine learning models"""
    # Create feature vector
    feature = dice.copy()
    feature.extend([1 if cat in available_categories else 0 for cat in ALL_CATEGORIES])
    feature.append(roll_num)
    feature_gpu = cudf.DataFrame([feature])
    
    if roll_num < 3:  # Decide which dice to keep
        # Predict whether to keep each die
        keep_probs = [model.predict(feature_gpu).to_numpy()[0] for model in keep_models]
        
        # Keep dice with probability > 0.5
        keep_indices = [i for i, prob in enumerate(keep_probs) if prob > 0.5]
        
        return keep_indices
    
    else:  # Decide which category to score
        # Predict the best category index
        category_index = int(category_model.predict(feature_gpu).to_numpy()[0] + 0.5)  # Round to nearest integer
        
        # Make sure the category is available
        while ALL_CATEGORIES[category_index] not in available_categories:
            # If not available, find the next best category
            category_index = (category_index + 1) % len(ALL_CATEGORIES)
        
        return ALL_CATEGORIES[category_index]