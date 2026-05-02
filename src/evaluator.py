import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class ModelEvaluator:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = model.predict(X_test)

    def create_plots(self):
        print("\nCreating visualizations...")

        fig = plt.figure(figsize=(16, 10))

        # Plot 1: Predictions vs Actual
        ax1 = plt.subplot(2, 2, 1)
        dates = self.y_test.index
        plt.plot(dates, self.y_test.values, label='Actual', linewidth=2, color='blue')
        plt.plot(dates, self.predictions, label='Predicted', linewidth=2, alpha=0.7, color='red')
        plt.title('Actual vs Predicted EUR/USD', fontweight='bold', fontsize=13)
        plt.xlabel('Date')
        plt.ylabel('EUR/USD Rate')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Plot 2: Scatter plot with R²
        ax2 = plt.subplot(2, 2, 2)
        plt.scatter(self.y_test, self.predictions, alpha=0.5, s=30)
        min_val = min(self.y_test.min(), self.predictions.min())
        max_val = max(self.y_test.max(), self.predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_test, self.predictions)
        plt.title(f'Prediction Accuracy (R² = {r2:.4f})', fontweight='bold', fontsize=13)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        ax3 = plt.subplot(2, 2, 3)
        errors = self.y_test.values - self.predictions
        plt.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.title(f'Error Distribution (Mean={errors.mean():.6f})', fontweight='bold', fontsize=13)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Plot 4: Top feature importance
        ax4 = plt.subplot(2, 2, 4)
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.X_test.columns
            ).sort_values(ascending=False).head(12)

            colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
            plt.barh(range(len(importance)), importance.values, color=colors)
            plt.yticks(range(len(importance)), importance.index, fontsize=9)
            plt.title('Top 12 Feature Importances', fontweight='bold', fontsize=13)
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('evaluation_report.png', dpi=200, bbox_inches='tight')
        plt.show()

        print("✓ Plots saved to 'evaluation_report.png'")