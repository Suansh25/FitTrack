import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading Data...")
df = pd.read_csv('assets/train.csv')

df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Intensity_Factor'] = df['Duration'] * df['Heart_Rate']
df['Sex_encoded'] = df['Sex'].map({'male': 1, 'female': 0})

features = ['Sex_encoded', 'Age', 'Height', 'Weight', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Intensity_Factor']
X = df[features]
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Initializing Ensemble (XGBoost + LightGBM + CatBoost)...")

xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42)
lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=42)
cat = CatBoostRegressor(n_estimators=500, learning_rate=0.05, depth=8, verbose=0, random_state=42)


ensemble_model = VotingRegressor(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    weights=[1, 1, 1] 
)

start_time = time.time()
print("Training the Ensemble Model")
ensemble_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f" Training Complete in {elapsed_time:.2f} seconds.")


print("Evaluating Performance of the models")
y_pred = ensemble_model.predict(X_test)

metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R2 Score": r2_score(y_test, y_pred)
}

for k, v in metrics.items():
    print(f"{k}: {v:.5f}")


print(" Generating Research Visuals")
plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.2, s=1, color='#2c3e50')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Prediction Reliability (Actual vs Predicted)')
plt.xlabel('Actual Calories')
plt.ylabel('Ensemble Predictions')


plt.subplot(1, 3, 2)
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, alpha=0.1, s=1, color='teal')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Analysis')
plt.xlabel('Predicted Value')
plt.ylabel('Residual Error')


plt.subplot(1, 3, 3)

non_zero_mask = y_test > 0
rel_error = (np.abs(residuals[non_zero_mask]) / y_test[non_zero_mask]) * 100
sns.histplot(rel_error, bins=50, kde=True, color='orange')
plt.title('Distribution of Relative Error (%)')
plt.xlabel('Percentage Error')

plt.tight_layout()
plt.show()