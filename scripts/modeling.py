"""
modeling.py - Spark MLlib code for building, training, and evaluating models
Handles classification model training, evaluation, and persistence
"""

import time
from typing import Dict, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier,
    LogisticRegressionModel, RandomForestClassificationModel, LinearSVCModel, GBTClassificationModel
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator
)
from pyspark.sql import functions as F


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIGS = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {"maxIter": 50, "regParam": 0.01}
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {"numTrees": 100, "maxDepth": 10, "seed": 42}
    },
    "LinearSVM": {
        "class": LinearSVC,
        "params": {"maxIter": 20, "regParam": 0.1}
    },
    "GBT": {
        "class": GBTClassifier,
        "params": {"maxIter": 20, "seed": 42}
    }
}


# ============================================================================
# Model Training
# ============================================================================

def train_model(train_df: DataFrame, model_name: str, 
                model_params: Dict = None) -> Union[LogisticRegressionModel,
                                                     RandomForestClassificationModel,
                                                     LinearSVCModel,
                                                     GBTClassificationModel]:
    """
    Train a classification model on the training data.
    
    Args:
        train_df: Training DataFrame with 'features' and 'label' columns
        model_name: Name of the model to train
        model_params: Optional override for default parameters
        
    Returns:
        Trained Spark ML model
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    params = model_params or config["params"]
    model_class = config["class"]
    
    model = model_class(featuresCol="features", labelCol="label", **params)
    trained_model = model.fit(train_df)
    
    return trained_model


def train_all_models(train_df: DataFrame, val_df: DataFrame) -> Dict[str, Dict]:
    """
    Train all configured models and return evaluation metrics.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        
    Returns:
        Dictionary of model results
    """
    results = {}
    
    for model_name in MODEL_CONFIGS.keys():
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        model = train_model(train_df, model_name)
        train_time = time.time() - start_time
        
        metrics = evaluate_model(model, val_df, model_name)
        metrics["train_time_sec"] = round(train_time, 2)
        results[model_name] = metrics
    
    return results


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(model, eval_df: DataFrame, model_name: str = "") -> Dict:
    """
    Evaluate a trained model on a DataFrame.
    
    Args:
        model: Trained Spark ML model
        eval_df: Evaluation DataFrame with 'features' and 'label' columns
        model_name: Name prefix for logging
        
    Returns:
        Dictionary with evaluation metrics (accuracy, f1, auc)
    """
    predictions = model.transform(eval_df)
    
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    
    accuracy = acc_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)
    
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4),
        "auc": round(auc, 4),
        "predictions": predictions
    }
    
    if model_name:
        print(f"{model_name:<22s}  acc={accuracy:.4f}  f1={f1_score:.4f}  AUC={auc:.4f}")
    
    return metrics


def get_confusion_matrix(predictions: DataFrame) -> DataFrame:
    """
    Generate confusion matrix from model predictions.
    
    Args:
        predictions: DataFrame with 'label' and 'prediction' columns
        
    Returns:
        Pandas DataFrame with confusion matrix counts
    """
    cm = (predictions
          .groupBy("label", "prediction")
          .count()
          .toPandas()
          .pivot(index="label", columns="prediction", values="count")
          .fillna(0)
          .astype(int))
    
    return cm


# ============================================================================
# Model Persistence
# ============================================================================

def save_model(model, model_path: str):
    """
    Save trained model to disk.
    
    Args:
        model: Trained Spark ML model
        model_path: Destination path
    """
    import shutil
    shutil.rmtree(model_path, ignore_errors=True)
    model.write().overwrite().save(model_path)
    print(f"Model saved to: {model_path}")


def save_pipeline(pipeline_model, pipeline_path: str):
    """
    Save feature pipeline to disk.
    
    Args:
        pipeline_model: Fitted PipelineModel
        pipeline_path: Destination path
    """
    import shutil
    shutil.rmtree(pipeline_path, ignore_errors=True)
    pipeline_model.write().overwrite().save(pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")