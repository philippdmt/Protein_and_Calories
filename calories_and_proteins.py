import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# read Dataset used for machine learning
calories = pd.read_csv("C:/Users/phili/Documents/HSG/Computer Science/Gruppenarbeit/calories.csv")


def determine_training_type(Heart_Rate,Age):
    if Heart_Rate >= 0.6 * (220 - Age):  # 60% of maximum heart frequency
        return "Cardio"
    else:
        return "Kraft"

calories['Training_Type'] = calories.apply(lambda row: determine_training_type(row['Heart_Rate'], row['Age']), axis=1)

# target variable
y = calories["Calories"]

# choose features
features = calories.drop(columns=['User_ID', 'Heart_Rate', 'Body_Temp','Calories'])


# change categorical values in dummy variables
categorical_values = ['Gender','Training_Type']
X = pd.get_dummies(features, columns=categorical_values, drop_first=False)  # WICHTIG: drop_first=False

# Split up in testset and trainingsset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Try a linear regression for machine learning
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predict the test set
y_pred = linreg.predict(X_test)

# Look up whether the results are reliable
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mean_calories = np.mean(y)

#Fill in the datas
gender = "Male"
training = "Cardio"
goal = "Bulk"
person = {
    "Age": 19,
    "Duration": 60,
    "Weight": 80,
    "Height": 188,
    "Gender_Female": 1 if gender.lower() == "female" else 0,
    "Gender_Male": 1 if gender.lower() == "male" else 0,
    "Training_Type_Cardio": 1 if training.lower() == "cardio" else 0,
    "Training_Type_Kraft": 1 if training.lower() == "kraft" else 0,
    "Goal_Cut": 1 if goal.lower() == "cut" else 0,
    "Goal_Maintain": 1 if goal.lower() == "maintain" else 0,
    "Goal_Bulk": 1 if goal.lower() == "bulk" else 0,
}

person_df = pd.DataFrame([person])

# Sicherstellen, dass ALLE Spalten wie im Trainingsdatensatz vorhanden sind
person_df = person_df.reindex(columns=X.columns, fill_value=0)

# Vorhersage
calorie_prediction_training = float(linreg.predict(person_df)[0])

print("Vorhergesagte Kalorien:", calorie_prediction_training)

def grundumsatz(age, weight, height, gender): 
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("gender muss 'male' oder 'female' sein")
    
    return bmr

calorie_prediction_bmr = grundumsatz(
    age=person["Age"],
    weight=person["Weight"],
    height=person["Height"],
    gender=gender
)
calorie_prediction_bmr

Total_burned_calories = calorie_prediction_bmr + calorie_prediction_training

def proteinbedarf(weight, goal):
    goal = goal.lower()

    if goal == "cut":         # Abnehmen
        factor = 2.0          # Mittelwert aus 1.8–2.4
    elif goal == "maintain":  # Gewicht halten
        factor = 1.6          # Mittelwert aus 1.4–1.8
    elif goal == "bulk":      # Muskelaufbau / Zunehmen
        factor = 1.9          # Mittelwert aus 1.6–2.2
    else:
        raise ValueError("goal muss 'cut', 'maintain' oder 'bulk' sein")

    return weight * factor

protein_person = proteinbedarf(person["Weight"],goal)