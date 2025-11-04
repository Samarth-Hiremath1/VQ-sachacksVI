import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Any
import logging
import os
from datetime import datetime

from app.models.body_language_classifier import PostureClassifier, GestureRecognizer
from app.services.mlflow_tracker import MLflowTracker, ModelTrainingTracker
from app.services.feature_extractor import PoseFeatureExtractor


class ModelTrainer:
    """Training pipeline for PyTorch body language models with MLflow tracking"""
    
    def __init__(self, 
                 mlflow_tracking_uri: str = None,
                 experiment_name: str = "body-language-training"):
        """
        Initialize model trainer
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow tracking
        self.mlflow_tracker = MLflowTracker(mlflow_tracking_uri, experiment_name)
        self.training_tracker = ModelTrainingTracker(self.mlflow_tracker)
        
        # Initialize feature extractor
        self.feature_extractor = PoseFeatureExtractor()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def generate_synthetic_data(self, 
                              num_samples: int = 1000,
                              sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for demonstration
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of pose sequences
            
        Returns:
            Tuple of (posture_data, posture_labels, sequence_data, gesture_labels)
        """
        self.logger.info(f"Generating {num_samples} synthetic training samples...")
        
        # Generate synthetic pose landmarks (33 landmarks x 3 coordinates)
        posture_data = []
        posture_labels = []
        sequence_data = []
        gesture_labels = []
        
        for i in range(num_samples):
            # Generate random pose landmarks
            landmarks = np.random.randn(33, 3) * 0.1
            
            # Add some structure to make it more realistic
            # Shoulders
            landmarks[11] = [-0.2, 0.0, 0.0]  # Left shoulder
            landmarks[12] = [0.2, 0.0, 0.0]   # Right shoulder
            
            # Hips
            landmarks[23] = [-0.1, -0.5, 0.0]  # Left hip
            landmarks[24] = [0.1, -0.5, 0.0]   # Right hip
            
            # Add noise and variations for different posture classes
            posture_class = i % 3  # 0: poor, 1: fair, 2: good
            
            if posture_class == 0:  # Poor posture
                landmarks[0, 0] += 0.1  # Forward head
                landmarks[11, 1] += 0.05  # Uneven shoulders
            elif posture_class == 2:  # Good posture
                landmarks[0, 0] -= 0.02  # Aligned head
                landmarks[11, 1] = landmarks[12, 1]  # Even shoulders
            
            posture_data.append(landmarks.flatten())
            posture_labels.append(posture_class)
            
            # Generate sequence data for gesture recognition
            sequence = []
            gesture_class = i % 5  # 5 gesture classes
            
            for frame in range(sequence_length):
                frame_landmarks = landmarks.copy()
                
                # Add temporal variations based on gesture class
                if gesture_class == 1:  # Pointing
                    frame_landmarks[16, 0] += 0.3 * np.sin(frame * 0.2)  # Right wrist movement
                elif gesture_class == 2:  # Open palm
                    frame_landmarks[15, 1] += 0.2 * np.cos(frame * 0.1)  # Left wrist up
                    frame_landmarks[16, 1] += 0.2 * np.cos(frame * 0.1)  # Right wrist up
                elif gesture_class == 3:  # Crossed arms
                    frame_landmarks[15, 0] += 0.2  # Left wrist to right
                    frame_landmarks[16, 0] -= 0.2  # Right wrist to left
                
                sequence.append(frame_landmarks.flatten())
            
            sequence_data.append(sequence)
            gesture_labels.append(gesture_class)
        
        return (
            np.array(posture_data, dtype=np.float32),
            np.array(posture_labels, dtype=np.int64),
            np.array(sequence_data, dtype=np.float32),
            np.array(gesture_labels, dtype=np.int64)
        )
    
    def train_posture_classifier(self, 
                               training_config: Dict[str, Any] = None) -> PostureClassifier:
        """
        Train posture classifier model
        
        Args:
            training_config: Training configuration parameters
            
        Returns:
            Trained posture classifier
        """
        if training_config is None:
            training_config = {
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_epochs": 50,
                "weight_decay": 1e-4,
                "dropout_rate": 0.3
            }
        
        self.logger.info("Starting posture classifier training...")
        
        # Generate training data
        posture_data, posture_labels, _, _ = self.generate_synthetic_data(num_samples=2000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            posture_data, posture_labels, test_size=0.2, random_state=42, stratify=posture_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=training_config["batch_size"], shuffle=False)
        
        # Initialize model
        model = PostureClassifier(input_dim=99, hidden_dim=128, num_classes=3).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(), 
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Training loop
        for epoch in range(training_config["num_epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                class_logits, confidence = model(batch_X)
                loss = criterion(class_logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    class_logits, confidence = model(batch_X)
                    loss = criterion(class_logits, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(class_logits.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Store history
            training_history["train_loss"].append(train_loss_avg)
            training_history["train_accuracy"].append(train_accuracy)
            training_history["val_loss"].append(val_loss_avg)
            training_history["val_accuracy"].append(val_accuracy)
            
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{training_config['num_epochs']}: "
                    f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}"
                )
        
        # Final evaluation on test set
        model.eval()
        test_predictions = []
        test_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                class_logits, confidence = model(batch_X)
                _, predicted = torch.max(class_logits.data, 1)
                
                test_predictions.extend(predicted.cpu().numpy())
                test_true.extend(batch_y.cpu().numpy())
        
        # Calculate final metrics
        test_accuracy = accuracy_score(test_true, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, test_predictions, average='weighted'
        )
        
        final_metrics = {
            "test_accuracy": test_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }
        
        # Track training with MLflow
        run_id = self.training_tracker.track_posture_classifier_training(
            model=model,
            training_config=training_config,
            training_history=training_history,
            final_metrics=final_metrics
        )
        
        self.logger.info(f"Posture classifier training completed. MLflow run ID: {run_id}")
        self.logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        
        return model
    
    def train_gesture_recognizer(self, 
                               training_config: Dict[str, Any] = None) -> GestureRecognizer:
        """
        Train gesture recognizer model
        
        Args:
            training_config: Training configuration parameters
            
        Returns:
            Trained gesture recognizer
        """
        if training_config is None:
            training_config = {
                "batch_size": 16,
                "learning_rate": 0.0005,
                "num_epochs": 40,
                "weight_decay": 1e-4,
                "sequence_length": 30
            }
        
        self.logger.info("Starting gesture recognizer training...")
        
        # Generate training data
        _, _, sequence_data, gesture_labels = self.generate_synthetic_data(
            num_samples=1500, 
            sequence_length=training_config["sequence_length"]
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequence_data, gesture_labels, test_size=0.2, random_state=42, stratify=gesture_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=training_config["batch_size"], shuffle=False)
        
        # Initialize model
        model = GestureRecognizer(
            sequence_length=training_config["sequence_length"],
            input_dim=99,
            hidden_dim=128
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Training loop
        for epoch in range(training_config["num_epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                gesture_logits, movement_intensity = model(batch_X)
                loss = criterion(gesture_logits, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(gesture_logits.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    gesture_logits, movement_intensity = model(batch_X)
                    loss = criterion(gesture_logits, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(gesture_logits.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Store history
            training_history["train_loss"].append(train_loss_avg)
            training_history["train_accuracy"].append(train_accuracy)
            training_history["val_loss"].append(val_loss_avg)
            training_history["val_accuracy"].append(val_accuracy)
            
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{training_config['num_epochs']}: "
                    f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}"
                )
        
        # Final evaluation on test set
        model.eval()
        test_predictions = []
        test_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                gesture_logits, movement_intensity = model(batch_X)
                _, predicted = torch.max(gesture_logits.data, 1)
                
                test_predictions.extend(predicted.cpu().numpy())
                test_true.extend(batch_y.cpu().numpy())
        
        # Calculate final metrics
        test_accuracy = accuracy_score(test_true, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, test_predictions, average='weighted'
        )
        
        final_metrics = {
            "test_accuracy": test_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }
        
        # Track training with MLflow
        run_id = self.training_tracker.track_gesture_recognizer_training(
            model=model,
            training_config=training_config,
            training_history=training_history,
            final_metrics=final_metrics
        )
        
        self.logger.info(f"Gesture recognizer training completed. MLflow run ID: {run_id}")
        self.logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        
        return model


def main():
    """Main training script"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = ModelTrainer(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name="body-language-models"
    )
    
    # Train posture classifier
    posture_model = trainer.train_posture_classifier()
    
    # Train gesture recognizer
    gesture_model = trainer.train_gesture_recognizer()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()