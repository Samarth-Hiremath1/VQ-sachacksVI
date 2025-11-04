import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Any
import logging
import os
from datetime import datetime

from app.models.body_language_classifier import PostureClassifier, GestureRecognizer, BodyLanguageAnalyzer
from app.services.mlflow_tracker import MLflowTracker, ModelEvaluationTracker
from app.training.train_models import ModelTrainer


class ModelEvaluator:
    """Model evaluation pipeline with MLflow tracking"""
    
    def __init__(self, mlflow_tracking_uri: str = None):
        """
        Initialize model evaluator
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow tracking
        self.mlflow_tracker = MLflowTracker(
            mlflow_tracking_uri, 
            experiment_name="body-language-evaluation"
        )
        self.evaluation_tracker = ModelEvaluationTracker(self.mlflow_tracker)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def evaluate_posture_classifier(self, 
                                  model: PostureClassifier,
                                  test_data: np.ndarray,
                                  test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate posture classifier model
        
        Args:
            model: Trained posture classifier
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating posture classifier...")
        
        model.eval()
        predictions = []
        confidence_scores = []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                input_tensor = torch.FloatTensor(test_data[i]).unsqueeze(0).to(self.device)
                result = model.predict_with_confidence(input_tensor)
                
                predictions.append(result['predicted_class'][0].item())
                confidence_scores.append(result['confidence'][0].item())
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            test_labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_confidence": np.mean(confidence_scores),
            "min_confidence": np.min(confidence_scores),
            "max_confidence": np.max(confidence_scores),
            "confidence_std": np.std(confidence_scores)
        }
        
        # Add per-class metrics
        class_names = ["poor", "fair", "good"]
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f"precision_{class_name}"] = precision_per_class[i]
                metrics[f"recall_{class_name}"] = recall_per_class[i]
                metrics[f"f1_{class_name}"] = f1_per_class[i]
        
        self.logger.info(f"Posture classifier evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics, cm
    
    def evaluate_gesture_recognizer(self,
                                  model: GestureRecognizer,
                                  test_sequences: np.ndarray,
                                  test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate gesture recognizer model
        
        Args:
            model: Trained gesture recognizer
            test_sequences: Test sequence data
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating gesture recognizer...")
        
        model.eval()
        predictions = []
        confidence_scores = []
        movement_intensities = []
        
        with torch.no_grad():
            for i in range(len(test_sequences)):
                input_tensor = torch.FloatTensor(test_sequences[i]).unsqueeze(0).to(self.device)
                result = model.predict_gestures(input_tensor)
                
                predictions.append(result['predicted_gesture'][0].item())
                confidence_scores.append(torch.max(result['gesture_probabilities'][0]).item())
                movement_intensities.append(result['movement_intensity'][0].item())
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            test_labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_confidence": np.mean(confidence_scores),
            "min_confidence": np.min(confidence_scores),
            "max_confidence": np.max(confidence_scores),
            "confidence_std": np.std(confidence_scores),
            "avg_movement_intensity": np.mean(movement_intensities),
            "movement_intensity_std": np.std(movement_intensities)
        }
        
        # Add per-class metrics
        gesture_classes = ["neutral", "pointing", "open_palm", "crossed_arms", "hands_on_hips"]
        for i, gesture_name in enumerate(gesture_classes):
            if i < len(precision_per_class):
                metrics[f"precision_{gesture_name}"] = precision_per_class[i]
                metrics[f"recall_{gesture_name}"] = recall_per_class[i]
                metrics[f"f1_{gesture_name}"] = f1_per_class[i]
        
        self.logger.info(f"Gesture recognizer evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics, cm
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all models
        
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        # Generate test data
        trainer = ModelTrainer()
        posture_data, posture_labels, sequence_data, gesture_labels = trainer.generate_synthetic_data(
            num_samples=500
        )
        
        # Initialize models (in practice, these would be loaded from MLflow)
        posture_model = PostureClassifier(input_dim=99, hidden_dim=128, num_classes=3).to(self.device)
        gesture_model = GestureRecognizer(sequence_length=30, input_dim=99, hidden_dim=128).to(self.device)
        
        # Evaluate posture classifier
        posture_metrics, posture_cm = self.evaluate_posture_classifier(
            posture_model, posture_data, posture_labels
        )
        
        # Evaluate gesture recognizer
        gesture_metrics, gesture_cm = self.evaluate_gesture_recognizer(
            gesture_model, sequence_data, gesture_labels
        )
        
        # Create comprehensive evaluation report
        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "posture_classifier": {
                "metrics": posture_metrics,
                "confusion_matrix": posture_cm.tolist()
            },
            "gesture_recognizer": {
                "metrics": gesture_metrics,
                "confusion_matrix": gesture_cm.tolist()
            },
            "overall_performance": {
                "avg_accuracy": (posture_metrics["accuracy"] + gesture_metrics["accuracy"]) / 2,
                "avg_f1_score": (posture_metrics["f1_score"] + gesture_metrics["f1_score"]) / 2
            }
        }
        
        # Track evaluation with MLflow
        run_id = self.mlflow_tracker.start_run(
            run_name=f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"evaluation_type": "comprehensive", "models": "posture_classifier,gesture_recognizer"}
        )
        
        # Log posture classifier evaluation
        self.mlflow_tracker.log_model_evaluation(
            model_name="posture_classifier",
            evaluation_metrics=posture_metrics,
            confusion_matrix=posture_cm
        )
        
        # Log gesture recognizer evaluation
        self.mlflow_tracker.log_model_evaluation(
            model_name="gesture_recognizer",
            evaluation_metrics=gesture_metrics,
            confusion_matrix=gesture_cm
        )
        
        # Log overall metrics
        import mlflow
        mlflow.log_metrics({
            "overall_avg_accuracy": evaluation_results["overall_performance"]["avg_accuracy"],
            "overall_avg_f1_score": evaluation_results["overall_performance"]["avg_f1_score"]
        })
        
        # Log evaluation report
        mlflow.log_dict(evaluation_results, "comprehensive_evaluation_report.json")
        
        self.mlflow_tracker.end_run()
        
        self.logger.info(f"Comprehensive evaluation completed. MLflow run ID: {run_id}")
        return evaluation_results


def main():
    """Main evaluation script"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print("Evaluation completed successfully!")
    print(f"Overall average accuracy: {results['overall_performance']['avg_accuracy']:.4f}")
    print(f"Overall average F1 score: {results['overall_performance']['avg_f1_score']:.4f}")


if __name__ == "__main__":
    main()