import mlflow
import mlflow.pytorch
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime
import json

from app.models.body_language_classifier import PostureClassifier, GestureRecognizer, BodyLanguageAnalyzer


class MLflowTracker:
    """MLflow experiment tracking for PyTorch body language models"""
    
    def __init__(self, 
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "body-language-analysis"):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        self.logger = logging.getLogger(__name__)
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        else:
            # Default to local file store
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            self.experiment_id = self.experiment.experiment_id
            self.logger.info(f"Using MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
        except Exception as e:
            self.logger.error(f"Failed to set MLflow experiment: {e}")
            raise e
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"body_language_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name, tags=tags)
        run_id = mlflow.active_run().info.run_id
        self.logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
        return run_id
    
    def log_model_training(self,
                          model: torch.nn.Module,
                          model_name: str,
                          training_params: Dict[str, Any],
                          training_metrics: Dict[str, float],
                          validation_metrics: Dict[str, float],
                          model_artifacts: Optional[Dict[str, Any]] = None):
        """
        Log model training information to MLflow
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            training_params: Training hyperparameters
            training_metrics: Training metrics
            validation_metrics: Validation metrics
            model_artifacts: Additional model artifacts
        """
        try:
            # Log parameters
            for param_name, param_value in training_params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log training metrics
            for metric_name, metric_value in training_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            # Log validation metrics
            for metric_name, metric_value in validation_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            # Log model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"models/{model_name}",
                registered_model_name=f"{model_name}_registered"
            )
            
            # Log additional artifacts
            if model_artifacts:
                for artifact_name, artifact_data in model_artifacts.items():
                    if isinstance(artifact_data, dict):
                        # Save as JSON
                        artifact_path = f"{artifact_name}.json"
                        with open(artifact_path, 'w') as f:
                            json.dump(artifact_data, f, indent=2)
                        mlflow.log_artifact(artifact_path)
                        os.remove(artifact_path)  # Clean up temporary file
                    else:
                        # Save as text
                        artifact_path = f"{artifact_name}.txt"
                        with open(artifact_path, 'w') as f:
                            f.write(str(artifact_data))
                        mlflow.log_artifact(artifact_path)
                        os.remove(artifact_path)  # Clean up temporary file
            
            self.logger.info(f"Logged training information for model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model training: {e}")
            raise e
    
    def log_model_evaluation(self,
                           model_name: str,
                           evaluation_metrics: Dict[str, float],
                           confusion_matrix: Optional[np.ndarray] = None,
                           feature_importance: Optional[Dict[str, float]] = None):
        """
        Log model evaluation results
        
        Args:
            model_name: Name of the model
            evaluation_metrics: Evaluation metrics
            confusion_matrix: Optional confusion matrix
            feature_importance: Optional feature importance scores
        """
        try:
            # Log evaluation metrics
            for metric_name, metric_value in evaluation_metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            # Log confusion matrix if provided
            if confusion_matrix is not None:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                confusion_matrix_path = f"confusion_matrix_{model_name}.png"
                plt.savefig(confusion_matrix_path)
                mlflow.log_artifact(confusion_matrix_path)
                plt.close()
                os.remove(confusion_matrix_path)  # Clean up
            
            # Log feature importance if provided
            if feature_importance:
                importance_path = f"feature_importance_{model_name}.json"
                with open(importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)  # Clean up
            
            self.logger.info(f"Logged evaluation results for model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model evaluation: {e}")
            raise e
    
    def log_inference_metrics(self,
                            model_name: str,
                            inference_time: float,
                            batch_size: int,
                            input_shape: tuple,
                            confidence_scores: List[float]):
        """
        Log model inference performance metrics
        
        Args:
            model_name: Name of the model
            inference_time: Time taken for inference (seconds)
            batch_size: Batch size used
            input_shape: Shape of input data
            confidence_scores: List of confidence scores from predictions
        """
        try:
            # Log inference performance
            mlflow.log_metric(f"{model_name}_inference_time", inference_time)
            mlflow.log_metric(f"{model_name}_throughput", batch_size / inference_time)
            mlflow.log_param(f"{model_name}_batch_size", batch_size)
            mlflow.log_param(f"{model_name}_input_shape", str(input_shape))
            
            # Log confidence statistics
            if confidence_scores:
                mlflow.log_metric(f"{model_name}_avg_confidence", np.mean(confidence_scores))
                mlflow.log_metric(f"{model_name}_min_confidence", np.min(confidence_scores))
                mlflow.log_metric(f"{model_name}_max_confidence", np.max(confidence_scores))
                mlflow.log_metric(f"{model_name}_confidence_std", np.std(confidence_scores))
            
            self.logger.info(f"Logged inference metrics for model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log inference metrics: {e}")
    
    def register_model(self,
                      model: torch.nn.Module,
                      model_name: str,
                      model_version: str,
                      description: str,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a model in MLflow Model Registry
        
        Args:
            model: PyTorch model to register
            model_name: Name for the registered model
            model_version: Version of the model
            description: Description of the model
            tags: Optional tags for the model
            
        Returns:
            Model version URI
        """
        try:
            # Log the model first
            model_uri = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"models/{model_name}",
                registered_model_name=model_name
            ).model_uri
            
            # Create model version
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            model_version_obj = client.create_model_version(
                name=model_name,
                source=model_uri,
                description=description,
                tags=tags
            )
            
            self.logger.info(f"Registered model: {model_name} version {model_version_obj.version}")
            return model_uri
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise e
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> torch.nn.Module:
        """
        Load a model from MLflow Model Registry
        
        Args:
            model_name: Name of the registered model
            version: Version to load (latest if None)
            
        Returns:
            Loaded PyTorch model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pytorch.load_model(model_uri)
            self.logger.info(f"Loaded model: {model_name} version {version or 'latest'}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
    
    def compare_models(self, run_ids: List[str], metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple model runs
        
        Args:
            run_ids: List of MLflow run IDs to compare
            metrics: List of metric names to compare
            
        Returns:
            Dictionary with comparison results
        """
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            comparison_results = {}
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                run_metrics = {}
                
                for metric in metrics:
                    if metric in run.data.metrics:
                        run_metrics[metric] = run.data.metrics[metric]
                
                comparison_results[run_id] = run_metrics
            
            self.logger.info(f"Compared {len(run_ids)} model runs")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            raise e
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
            self.logger.info("Ended MLflow run")
        except Exception as e:
            self.logger.error(f"Failed to end MLflow run: {e}")


class ModelTrainingTracker:
    """Specialized tracker for model training workflows"""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        """
        Initialize model training tracker
        
        Args:
            mlflow_tracker: MLflow tracker instance
        """
        self.mlflow_tracker = mlflow_tracker
        self.logger = logging.getLogger(__name__)
    
    def track_posture_classifier_training(self,
                                        model: PostureClassifier,
                                        training_config: Dict[str, Any],
                                        training_history: Dict[str, List[float]],
                                        final_metrics: Dict[str, float]):
        """
        Track posture classifier training
        
        Args:
            model: Trained posture classifier
            training_config: Training configuration
            training_history: Training history with loss/accuracy per epoch
            final_metrics: Final evaluation metrics
        """
        try:
            run_id = self.mlflow_tracker.start_run(
                run_name=f"posture_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"model_type": "posture_classifier", "framework": "pytorch"}
            )
            
            # Log training configuration
            self.mlflow_tracker.log_model_training(
                model=model,
                model_name="posture_classifier",
                training_params=training_config,
                training_metrics={
                    "final_train_loss": training_history["train_loss"][-1],
                    "final_train_accuracy": training_history["train_accuracy"][-1]
                },
                validation_metrics={
                    "final_val_loss": training_history["val_loss"][-1],
                    "final_val_accuracy": training_history["val_accuracy"][-1]
                },
                model_artifacts={
                    "training_history": training_history,
                    "model_architecture": {
                        "input_dim": model.input_dim,
                        "hidden_dim": model.hidden_dim,
                        "num_classes": model.num_classes
                    }
                }
            )
            
            # Log final evaluation metrics
            self.mlflow_tracker.log_model_evaluation(
                model_name="posture_classifier",
                evaluation_metrics=final_metrics
            )
            
            # Log training curves
            for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(
                zip(training_history["train_loss"], training_history["val_loss"],
                    training_history["train_accuracy"], training_history["val_accuracy"])
            ):
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc
                }, step=epoch)
            
            self.mlflow_tracker.end_run()
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to track posture classifier training: {e}")
            self.mlflow_tracker.end_run()
            raise e
    
    def track_gesture_recognizer_training(self,
                                        model: GestureRecognizer,
                                        training_config: Dict[str, Any],
                                        training_history: Dict[str, List[float]],
                                        final_metrics: Dict[str, float]):
        """
        Track gesture recognizer training
        
        Args:
            model: Trained gesture recognizer
            training_config: Training configuration
            training_history: Training history
            final_metrics: Final evaluation metrics
        """
        try:
            run_id = self.mlflow_tracker.start_run(
                run_name=f"gesture_recognizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"model_type": "gesture_recognizer", "framework": "pytorch"}
            )
            
            # Log training configuration
            self.mlflow_tracker.log_model_training(
                model=model,
                model_name="gesture_recognizer",
                training_params=training_config,
                training_metrics={
                    "final_train_loss": training_history["train_loss"][-1],
                    "final_train_accuracy": training_history["train_accuracy"][-1]
                },
                validation_metrics={
                    "final_val_loss": training_history["val_loss"][-1],
                    "final_val_accuracy": training_history["val_accuracy"][-1]
                },
                model_artifacts={
                    "training_history": training_history,
                    "model_architecture": {
                        "sequence_length": model.sequence_length,
                        "input_dim": model.input_dim,
                        "hidden_dim": model.hidden_dim,
                        "num_gesture_classes": model.num_gesture_classes,
                        "gesture_classes": model.gesture_classes
                    }
                }
            )
            
            # Log final evaluation metrics
            self.mlflow_tracker.log_model_evaluation(
                model_name="gesture_recognizer",
                evaluation_metrics=final_metrics
            )
            
            # Log training curves
            for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(
                zip(training_history["train_loss"], training_history["val_loss"],
                    training_history["train_accuracy"], training_history["val_accuracy"])
            ):
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc
                }, step=epoch)
            
            self.mlflow_tracker.end_run()
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to track gesture recognizer training: {e}")
            self.mlflow_tracker.end_run()
            raise e


class ModelEvaluationTracker:
    """Specialized tracker for model evaluation and comparison"""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        """
        Initialize model evaluation tracker
        
        Args:
            mlflow_tracker: MLflow tracker instance
        """
        self.mlflow_tracker = mlflow_tracker
        self.logger = logging.getLogger(__name__)
    
    def create_model_comparison_report(self, 
                                     model_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comprehensive model comparison report
        
        Args:
            model_runs: List of model run information
            
        Returns:
            Comparison report
        """
        try:
            run_id = self.mlflow_tracker.start_run(
                run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"type": "model_comparison"}
            )
            
            # Extract run IDs and metrics
            run_ids = [run["run_id"] for run in model_runs]
            metrics_to_compare = ["val_accuracy", "val_loss", "eval_precision", "eval_recall", "eval_f1"]
            
            # Get comparison results
            comparison_results = self.mlflow_tracker.compare_models(run_ids, metrics_to_compare)
            
            # Find best model for each metric
            best_models = {}
            for metric in metrics_to_compare:
                best_run_id = None
                best_value = None
                
                for run_id, metrics in comparison_results.items():
                    if metric in metrics:
                        if best_value is None or (
                            metric == "val_loss" and metrics[metric] < best_value
                        ) or (
                            metric != "val_loss" and metrics[metric] > best_value
                        ):
                            best_value = metrics[metric]
                            best_run_id = run_id
                
                if best_run_id:
                    best_models[metric] = {"run_id": best_run_id, "value": best_value}
            
            # Create comparison report
            comparison_report = {
                "comparison_timestamp": datetime.now().isoformat(),
                "models_compared": len(run_ids),
                "metrics_compared": metrics_to_compare,
                "best_models": best_models,
                "detailed_results": comparison_results
            }
            
            # Log comparison report
            mlflow.log_dict(comparison_report, "model_comparison_report.json")
            
            self.mlflow_tracker.end_run()
            return comparison_report
            
        except Exception as e:
            self.logger.error(f"Failed to create model comparison report: {e}")
            self.mlflow_tracker.end_run()
            raise e