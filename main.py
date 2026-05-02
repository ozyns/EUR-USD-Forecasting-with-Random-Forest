from src.data_engineer import EnhancedDataEngineer
from src.model_developer import OptimizedModelDeveloper
from src.evaluator import ModelEvaluator
from datetime import datetime, timedelta

print("="*70)
print("EUR/USD FORECASTING — RANDOM FOREST")
print("="*70)

engineer = EnhancedDataEngineer(
    start_date=datetime.now() - timedelta(days=365*10),
    end_date=datetime.now()
)
processed_data = engineer.process_all()

developer = OptimizedModelDeveloper(processed_data)
developer.prepare_data()
developer.train_model()
results = developer.evaluate_model()
developer.save_model()

evaluator = ModelEvaluator(
    developer.model,
    developer.X_test,
    developer.y_test
)
evaluator.create_plots()

print("\n" + "="*70)
print("DONE!")
print("="*70)