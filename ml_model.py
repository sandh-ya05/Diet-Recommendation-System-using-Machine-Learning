import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import pickle
import os
import requests
from io import StringIO

# Path for saving/loading models and preprocessors
# Path for saving/loading models and preprocessors
MODELS_FILE = 'models.pkl'
DATASET_FILE = 'Food_and_Nutrition new.csv'

def load_nutrition_dataset():
    """Load the nutrition dataset from local file or fallback."""
    try:
        if os.path.exists(DATASET_FILE):
            df = pd.read_csv(DATASET_FILE)
            print("Loaded dataset from local CSV.")
            df = preprocess_nutrition_data(df)
        else:
            print(f"{DATASET_FILE} not found. Using fallback dataset.")
            df = create_fallback_dataset()
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return create_fallback_dataset()



def preprocess_nutrition_data(df):
    """Clean and preprocess the nutrition dataset."""
    print("Preprocessing nutrition data...")
    
    # Ensure all relevant columns are numeric, coercing errors
    numeric_columns = ['Ages', 'Height', 'Weight', 'Daily Calorie Target', 'Protein', 
                       'Sugar', 'Sodium', 'Calories', 'Carbohydrates', 'Fiber', 'Fat']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.lower().str.strip()
        df['Gender'] = df['Gender'].map({'female': 'female', 'male': 'male'}).fillna('male')
    
    if 'Activity Level' in df.columns:
        df['Activity Level'] = df['Activity Level'].str.lower().str.strip()
        activity_mapping = {
            'sedentary': 'sedentary', 'lightly active': 'light', 'light': 'light',
            'moderately active': 'moderate', 'moderate': 'moderate',
            'very active': 'active', 'active': 'active', 'extremely active': 'very_active'
        }
        df['Activity Level'] = df['Activity Level'].map(activity_mapping).fillna('moderate')
    
    if 'Dietary Preference' in df.columns:
        df['Dietary Preference'] = df['Dietary Preference'].str.lower().str.strip()
        diet_mapping = {
            'vegetarian': 'vegetarian', 'vegan': 'vegan', 'non-vegetarian': 'non_vegetarian',
            'pescatarian': 'pescatarian', 'gluten-free': 'gluten_free', 'gluten free': 'gluten_free'
        }
        df['Dietary Preference'] = df['Dietary Preference'].map(diet_mapping).fillna('non_vegetarian')
    
    if 'Height' in df.columns and 'Weight' in df.columns:
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    def get_body_type(bmi):
        if pd.isna(bmi): return 'normal'
        elif bmi < 18.5: return 'underweight'
        elif bmi < 25: return 'normal'
        elif bmi < 30: return 'overweight'
        else: return 'obese'
    
    df['Body_Type'] = df['BMI'].apply(get_body_type)
    
    if 'Fitness_Goal' not in df.columns and 'Disease' in df.columns:
        df['Disease'] = df['Disease'].str.lower().str.strip()
        goal_mapping = {
            'weight gain': 'weight_gain', 'weight loss': 'weight_loss', 'muscle gain': 'muscle_gain',
            'diabetes': 'maintenance', 'hypertension': 'maintenance', 'heart disease': 'maintenance',
            'general wellness': 'general_wellness'
        }
        df['Fitness_Goal'] = df['Disease'].map(goal_mapping).fillna('general_wellness')
    elif 'Fitness_Goal' not in df.columns:
        df['Fitness_Goal'] = 'general_wellness' # Default if neither is present
    
    df = df.dropna(subset=['Gender', 'Activity Level', 'Dietary Preference', 'Ages', 'Height', 'Weight', 'BMI'])
    
    meal_columns = ['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion']
    for col in meal_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Healthy balanced meal')
  
    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df

def create_fallback_dataset():
    """Create a fallback dataset if the CSV cannot be loaded."""
    print("Creating fallback dataset...")
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Gender': np.random.choice(['male', 'female'], n_samples),
        'Ages': np.random.randint(18, 70, n_samples),
        'Height': np.random.normal(170, 10, n_samples),
        'Weight': np.random.normal(70, 15, n_samples),
        'Activity Level': np.random.choice(['sedentary', 'light', 'moderate', 'active', 'very_active'], n_samples),
        'Dietary Preference': np.random.choice(['vegetarian', 'vegan', 'non_vegetarian', 'pescatarian'], n_samples),
        'Daily Calorie Target': np.random.randint(1200, 2500, n_samples),
        'Protein': np.random.randint(50, 200, n_samples),
        'Carbohydrates': np.random.randint(100, 300, n_samples),
        'Fat': np.random.randint(30, 100, n_samples),
        'Fiber': np.random.randint(15, 40, n_samples),
        'Sugar': np.random.randint(20, 100, n_samples),
        'Sodium': np.random.randint(1000, 3000, n_samples),
        'Breakfast Suggestion': ['Oatmeal with fresh fruits and nuts'] * n_samples,
        'Lunch Suggestion': ['Grilled chicken salad with mixed vegetables'] * n_samples,
        'Dinner Suggestion': ['Baked fish with quinoa and steamed broccoli'] * n_samples,
        'Snack Suggestion': ['Greek yogurt with berries and almonds'] * n_samples,
        'Fitness_Goal': np.random.choice(['weight_loss', 'muscle_gain', 'maintenance', 'general_wellness'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    def get_body_type(bmi):
        if bmi < 18.5: return 'underweight'
        elif bmi < 25: return 'normal'
        elif bmi < 30: return 'overweight'
        else: return 'obese'
    
    df['Body_Type'] = df['BMI'].apply(get_body_type)
    
    return df

def train_models():
    """Train machine learning models on the nutrition dataset."""
    df = load_nutrition_dataset()
    
    if df is None or df.empty:
        print("No dataset available for training.")
        return None
    
    print(f"Training models with dataset shape: {df.shape}")
    
    feature_columns = ['Ages', 'Height', 'Weight', 'BMI']
    
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing critical columns for training: {missing_cols}")
        # Attempt to use available columns if critical ones are missing
        available_features = [col for col in feature_columns if col in df.columns]
        if not available_features:
            print("No sufficient features for training.")
            return None
        feature_columns = available_features
        print(f"Training with available features: {feature_columns}")

    le_gender = LabelEncoder()
    le_activity = LabelEncoder()
    le_dietary = LabelEncoder()
    le_goal = LabelEncoder()
    
    X = df[feature_columns].copy()
    
    # Handle potential missing columns for encoding gracefully
    if 'Gender' in df.columns:
        X['gender_encoded'] = le_gender.fit_transform(df['Gender'])
    else:
        X['gender_encoded'] = 0 # Default or handle as needed
        print("Warning: 'Gender' column not found for encoding.")

    if 'Activity Level' in df.columns:
        X['activity_encoded'] = le_activity.fit_transform(df['Activity Level'])
    else:
        X['activity_encoded'] = 0
        print("Warning: 'Activity Level' column not found for encoding.")

    if 'Dietary Preference' in df.columns:
        X['dietary_encoded'] = le_dietary.fit_transform(df['Dietary Preference'])
    else:
        X['dietary_encoded'] = 0
        print("Warning: 'Dietary Preference' column not found for encoding.")
    
    y = le_goal.fit_transform(df['Fitness_Goal'])
    
    mask = ~(X.isna().any(axis=1) | pd.isna(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        print("No valid data for training after cleaning.")
        return None
    
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    knn_train_pred = knn.predict(X_train_scaled)
    knn_train_accuracy = accuracy_score(y_train, knn_train_pred)
    knn_train_f1 = f1_score(y_train, knn_train_pred, average='weighted', zero_division=0)

    knn_test_pred = knn.predict(X_test_scaled)
    knn_test_accuracy = accuracy_score(y_test, knn_test_pred)
    knn_test_f1 = f1_score(y_test, knn_test_pred, average='weighted', zero_division=0)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    rf_train_pred = rf.predict(X_train_scaled)
    rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
    rf_train_f1 = f1_score(y_train, rf_train_pred, average='weighted', zero_division=0)

    rf_test_pred = rf.predict(X_test_scaled)
    rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
    rf_test_f1 = f1_score(y_test, rf_test_pred, average='weighted', zero_division=0)
    
    print(f"\n--- Model Training & Testing Results ---")
    print(f"K-Nearest Neighbors (KNN):")
    print(f"  Training Accuracy: {knn_train_accuracy:.4f}")
    print(f"  Training F1-Score: {knn_train_f1:.4f}")
    print(f"  Testing Accuracy: {knn_test_accuracy:.4f}")
    print(f"  Testing F1-Score: {knn_test_f1:.4f}")
    print(f"\nRandom Forest:")
    print(f"  Training Accuracy: {rf_train_accuracy:.4f}")
    print(f"  Training F1-Score: {rf_train_f1:.4f}")
    print(f"  Testing Accuracy: {rf_test_accuracy:.4f}")
    print(f"  Testing F1-Score: {rf_test_f1:.4f}")
    print(f"----------------------------------------\n")
    
    models = {
        'knn': knn,
        'rf': rf,
        'scaler': scaler,
        'le_gender': le_gender,
        'le_activity': le_activity,
        'le_dietary': le_dietary,
        'le_goal': le_goal,
        'knn_train_accuracy': knn_train_accuracy,
        'knn_train_f1': knn_train_f1,
        'knn_test_accuracy': knn_test_accuracy,
        'knn_test_f1': knn_test_f1,
        'rf_train_accuracy': rf_train_accuracy,
        'rf_train_f1': rf_train_f1,
        'rf_test_accuracy': rf_test_accuracy,
        'rf_test_f1': rf_test_f1,
        'dataset': df # Store the preprocessed dataset with the models
    }
    
    with open(MODELS_FILE, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"Models trained and saved to {MODELS_FILE}!")
    
    return models

def load_models():
    """Load trained models or train new ones if not found."""
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, 'rb') as f:
                models = pickle.load(f)
            print(f"Models loaded from {MODELS_FILE}.")
            # Print all accuracies and metrics when loaded from file
            if 'knn_train_accuracy' in models and 'rf_train_accuracy' in models:
                print(f"\n--- Loaded Model Accuracies & Metrics ---")
                print(f"K-Nearest Neighbors (KNN):")
                print(f"  Training Accuracy: {models['knn_train_accuracy']:.4f}")
                print(f"  Training F1-Score: {models['knn_train_f1']:.4f}")
                print(f"  Testing Accuracy: {models['knn_test_accuracy']:.4f}")
                print(f"  Testing F1-Score: {models['knn_test_f1']:.4f}")
                print(f"\nRandom Forest:")
                print(f"  Training Accuracy: {models['rf_train_accuracy']:.4f}")
                print(f"  Training F1-Score: {models['rf_train_f1']:.4f}")
                print(f"  Testing Accuracy: {models['rf_test_accuracy']:.4f}")
                print(f"  Testing F1-Score: {models['rf_test_f1']:.4f}")
                print(f"-------------------------------------------\n")
            return models
        except Exception as e:
            print(f"Error loading models from {MODELS_FILE}: {e}. Retraining models.")
            return train_models()
    else:
        print(f"Models file {MODELS_FILE} not found. Training models.")
        return train_models()

def generate_personalized_diet_plan(models, user_data):
    """Generate personalized diet plan from the dataset based on user input."""
    df = models['dataset']
    
    user_bmi = user_data['weight'] / ((user_data['height'] / 100) ** 2)
    
    # Filter dataset based on user characteristics
    filtered_df = df[
        (df['Gender'] == user_data['gender']) &
        (df['Activity Level'] == user_data['activity_level']) &
        (df['Dietary Preference'] == user_data['food_preference'])
    ]
    
    # Broaden search if initial filter is too restrictive
    if filtered_df.empty:
        filtered_df = df[
            (df['Dietary Preference'] == user_data['food_preference']) &
            (abs(df['BMI'] - user_bmi) <= 5)
        ]
    
    if filtered_df.empty:
        filtered_df = df[df['Dietary Preference'] == user_data['food_preference']]
    
    if filtered_df.empty: # Fallback to a random sample if no matches
        filtered_df = df.sample(min(10, len(df)))
    
    # Get nutritional targets (averages from similar users)
    daily_calories = int(filtered_df['Daily Calorie Target'].mean()) if 'Daily Calorie Target' in filtered_df.columns and not filtered_df['Daily Calorie Target'].empty else 2000
    protein = round(filtered_df['Protein'].mean(), 1) if 'Protein' in filtered_df.columns and not filtered_df['Protein'].empty else 150
    carbs = round(filtered_df['Carbohydrates'].mean(), 1) if 'Carbohydrates' in filtered_df.columns and not filtered_df['Carbohydrates'].empty else 200
    fat = round(filtered_df['Fat'].mean(), 1) if 'Fat' in filtered_df.columns and not filtered_df['Fat'].empty else 65
    fiber = round(filtered_df['Fiber'].mean(), 1) if 'Fiber' in filtered_df.columns and not filtered_df['Fiber'].empty else 25
    sugar = round(filtered_df['Sugar'].mean(), 1) if 'Sugar' in filtered_df.columns and not filtered_df['Sugar'].empty else 50
    sodium = round(filtered_df['Sodium'].mean(), 1) if 'Sodium' in filtered_df.columns and not filtered_df['Sodium'].empty else 2000
    
    # Get meal suggestions (most common from similar users)
    breakfast = filtered_df['Breakfast Suggestion'].mode().iloc[0] if 'Breakfast Suggestion' in filtered_df.columns and not filtered_df['Breakfast Suggestion'].mode().empty else "Oatmeal with fresh fruits and nuts"
    lunch = filtered_df['Lunch Suggestion'].mode().iloc[0] if 'Lunch Suggestion' in filtered_df.columns and not filtered_df['Lunch Suggestion'].mode().empty else "Grilled chicken salad with mixed vegetables"
    dinner = filtered_df['Dinner Suggestion'].mode().iloc[0] if 'Dinner Suggestion' in filtered_df.columns and not filtered_df['Dinner Suggestion'].mode().empty else "Baked fish with quinoa and steamed broccoli"
    snack = filtered_df['Snack Suggestion'].mode().iloc[0] if 'Snack Suggestion' in filtered_df.columns and not filtered_df['Snack Suggestion'].mode().empty else "Greek yogurt with berries and almonds"
    
    return {
        'daily_calories': daily_calories,
        'protein': protein,
        'carbohydrates': carbs,
        'fat': fat,
        'fiber': fiber,
        'sugar': sugar,
        'sodium': sodium,
        'breakfast': breakfast,
        'lunch': lunch,
        'dinner': dinner,
        'snack': snack
    }

if __name__ == '__main__':
    # This block will run when ml_model.py is executed directly
    # It will load/train models and print accuracy
    models = load_models()
    if models:
        print("\nML Model operations complete.")
