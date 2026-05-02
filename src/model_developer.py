import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class OptimizedModelDeveloper:

    def __init__(self, data):
        self.data = data
        self.model = None

    def prepare_data(self, test_size=0.2):
        print("Preparing data...")

        X = self.data.drop('EUR_USD', axis=1)
        y = self.data['EUR_USD']

        split_idx = int(len(X) * (1 - test_size))
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        print(f"✓ Train: {len(self.X_train)} samples")
        print(f"✓ Test: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        print("\nTraining Random Forest (optimized settings)...")

        # Random Forest with good hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=200,        # Number of trees in the forest
            max_depth=15,            # Maximum depth of trees
            min_samples_split=10,    # Minimum samples to split a node
            min_samples_leaf=4,      # Minimum samples in leaf node
            max_features='sqrt',     # Number of features to consider for splits
            bootstrap=True,          # Use bootstrap samples
            n_jobs=-1,              # Use all CPU cores for parallel processing
            random_state=42
        )

        self.model.fit(self.X_train, self.y_train)
        print("✓ Model trained!")

        return self.model

    def evaluate_model(self):
        print("\nEvaluating model...")

        y_pred = self.model.predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100

        # Additional metric: directional accuracy
        actual_direction = np.sign(self.y_test.diff())
        pred_direction = np.sign(pd.Series(y_pred, index=self.y_test.index).diff())
        directional_acc = (actual_direction == pred_direction).sum() / len(actual_direction) * 100

        print(f"\n{'='*50}")
        print(f"RANDOM FOREST MODEL PERFORMANCE:")
        print(f"{'='*50}")
        print(f"RMSE:                {rmse:.6f}")
        print(f"MAE:                 {mae:.6f}")
        print(f"R²:                  {r2:.6f}")
        print(f"MAPE:                {mape:.4f}%")
        print(f"Directional Accuracy: {directional_acc:.2f}%")
        print(f"{'='*50}")

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'directional_accuracy': directional_acc,
            'predictions': y_pred
        }

    def save_model(self):
        joblib.dump(self.model, 'trained_model.pkl')
        print("✓ Model saved as 'trained_model.pkl'")
