# 📦 Install dependencies
# !pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# 🔧 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam

# 📥 Load and Prepare Data
df = pd.read_csv('insurance.csv')
df.rename(columns={'expenses': 'target'}, inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

# 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔄 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🧠 Train Traditional Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    results[name] = {
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

# 📈 CNN for Tabular Data
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

cnn_model = Sequential([
    Input(shape=(X_train_scaled.shape[1], 1)),
    Conv1D(32, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output for regression
])

cnn_model.compile(optimizer=Adam(), loss='mse')
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=16, verbose=0)

cnn_preds = cnn_model.predict(X_test_cnn).flatten()
results["CNN"] = {
    "MSE": mean_squared_error(y_test, cnn_preds),
    "RMSE": np.sqrt(mean_squared_error(y_test, cnn_preds)),
    "MAE": mean_absolute_error(y_test, cnn_preds),
    "R2": r2_score(y_test, cnn_preds)
}

# 📊 Visualize Metrics
metrics_df = pd.DataFrame(results).T
metrics_df[['RMSE', 'MAE', 'R2']].plot(kind='bar', figsize=(10,6), title="Model Performance on Insurance Expenses")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🧾 Show results
print(metrics_df.round(2))
