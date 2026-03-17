import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import time
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score


print("Loading data (100k rows)...")
df = pd.read_csv('assets/train.csv')

df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Intensity_Factor'] = df['Duration'] * df['Heart_Rate']
df['Sex_encoded'] = df['Sex'].map({'male': 1, 'female': 0})

features = ['Sex_encoded', 'Age', 'Height', 'Weight', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Intensity_Factor']
X = df[features]
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_shard = X_train.sample(50000, random_state=42)
y_shard = y_train.loc[X_shard.index]


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300), 
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
        'tree_method': 'hist', 
        'device': 'cpu' 
    }
    model = XGBRegressor(**param, random_state=42)

    return -cross_val_score(model, X_shard, y_shard, cv=2, scoring='neg_mean_absolute_error').mean()

print("Tuning XGBoost...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) 


print("Training Final Ensemble of XGBoost, CatBoost and LightGBM on full rows (This will take time)...")
best_xgb = XGBRegressor(**study.best_params, random_state=42)
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.07, n_jobs=-1, random_state=42)
cat = CatBoostRegressor(n_estimators=300, learning_rate=0.07, verbose=0, random_state=42)

ensemble_model = VotingRegressor(estimators=[('xgb', best_xgb), ('lgbm', lgbm), ('cat', cat)])
start = time.time()
ensemble_model.fit(X_train, y_train)
print(f"Training Complete in {time.time()-start:.2f}s")



best_xgb.fit(X_train, y_train)
explainer = shap.TreeExplainer(best_xgb)


joblib.dump(ensemble_model, 'calories_model.pkl')
joblib.dump(explainer, 'shap_explainer.pkl')
joblib.dump(features, 'feature_names.pkl')
print("All models saved.")