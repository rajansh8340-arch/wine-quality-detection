import json
import logging
from src.data import get_train_test_data, preprocess_data, load_data
from src.model import WineQualityModelPipeline
from src.config import MODEL_PATH, METRICS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Wine Quality Model Training Pipeline...")
    
    # 1. Load and Split Data
    X_train, X_test, y_train, y_test = get_train_test_data()
    logger.info(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples.")
    logger.info(f"Positive (Good Quality) ratio in Train: {y_train.mean()*100:.2f}% | Test: {y_test.mean()*100:.2f}%")

    # 2. Cross-Validation
    pipeline = WineQualityModelPipeline(n_estimators=150, max_depth=12)
    full_df = load_data()
    X_full, y_full = preprocess_data(full_df)
    logger.info("Executing 5-Fold Stratified Cross-Validation...")
    cv_summary = pipeline.cross_validate(X_full, y_full)
    for metric, val in cv_summary.items():
        logger.info(f"  - {metric}: {val:.4f}")

    # 3. Train & Test Evaluation
    logger.info("Training on full train split and evaluating on hold-out test set...")
    metrics = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    metrics["cross_validation"] = cv_summary
    
    logger.info(f"Hold-out Test Accuracy:  {metrics['accuracy']*100:.2f}%")
    logger.info(f"Hold-out Test Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"Hold-out Test Recall:    {metrics['recall']*100:.2f}%")
    logger.info(f"Hold-out Test F1-Score:  {metrics['f1_score']*100:.2f}%")
    logger.info(f"Hold-out Test ROC-AUC:   {metrics['roc_auc']:.4f}")

    # 4. Save Artifacts
    pipeline.save(MODEL_PATH, METRICS_PATH)
    logger.info(f"Saved serialized model to: {MODEL_PATH}")
    logger.info(f"Saved evaluation metrics to: {METRICS_PATH}")
    logger.info("Training Pipeline Complete!")

if __name__ == "__main__":
    main()
