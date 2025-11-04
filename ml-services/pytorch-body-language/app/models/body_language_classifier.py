import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PostureClassifier(nn.Module):
    """PyTorch neural network for posture classification (good/poor posture scoring)"""
    
    def __init__(self, input_dim: int = 99, hidden_dim: int = 128, num_classes: int = 3):
        """
        Initialize posture classifier
        
        Args:
            input_dim: Input dimension (33 landmarks * 3 coordinates = 99)
            hidden_dim: Hidden layer dimension
            num_classes: Number of posture classes (poor=0, fair=1, good=2)
        """
        super(PostureClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (class_logits, confidence_scores)
        """
        features = self.feature_extractor(x)
        
        # Classification output
        class_logits = self.classifier(features)
        
        # Confidence output
        confidence = self.confidence_head(features)
        
        return class_logits, confidence
    
    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict posture class with confidence score
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            class_logits, confidence = self.forward(x)
            probabilities = F.softmax(class_logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            return {
                'predicted_class': predicted_classes,
                'probabilities': probabilities,
                'confidence': confidence.squeeze(),
                'posture_score': probabilities[:, 2]  # Good posture probability
            }


class GestureRecognizer(nn.Module):
    """PyTorch neural network for gesture recognition (hand movements and body positioning)"""
    
    def __init__(self, sequence_length: int = 30, input_dim: int = 99, hidden_dim: int = 128):
        """
        Initialize gesture recognizer with LSTM for temporal modeling
        
        Args:
            sequence_length: Length of pose sequence
            input_dim: Input dimension per frame (33 landmarks * 3 coordinates = 99)
            hidden_dim: Hidden dimension for LSTM
        """
        super(GestureRecognizer, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gesture classes: neutral, pointing, open_palm, crossed_arms, hands_on_hips
        self.gesture_classes = ['neutral', 'pointing', 'open_palm', 'crossed_arms', 'hands_on_hips']
        self.num_gesture_classes = len(self.gesture_classes)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional LSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification layers
        self.gesture_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_gesture_classes)
        )
        
        # Movement intensity estimation
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for gesture recognition
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Tuple of (gesture_logits, movement_intensity)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(attended_out, dim=1)
        
        # Gesture classification
        gesture_logits = self.gesture_classifier(pooled_features)
        
        # Movement intensity
        movement_intensity = self.movement_head(pooled_features)
        
        return gesture_logits, movement_intensity
    
    def predict_gestures(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict gestures with confidence and movement analysis
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with gesture predictions and movement analysis
        """
        self.eval()
        with torch.no_grad():
            gesture_logits, movement_intensity = self.forward(x)
            gesture_probs = F.softmax(gesture_logits, dim=1)
            predicted_gestures = torch.argmax(gesture_probs, dim=1)
            
            return {
                'predicted_gesture': predicted_gestures,
                'gesture_probabilities': gesture_probs,
                'movement_intensity': movement_intensity.squeeze(),
                'gesture_names': [self.gesture_classes[idx] for idx in predicted_gestures.cpu().numpy()]
            }


class TemporalSmoother:
    """Temporal smoothing for pose sequences to reduce noise and improve stability"""
    
    def __init__(self, window_size: int = 5, alpha: float = 0.7):
        """
        Initialize temporal smoother
        
        Args:
            window_size: Size of smoothing window
            alpha: Exponential smoothing factor
        """
        self.window_size = window_size
        self.alpha = alpha
        self.pose_history: List[np.ndarray] = []
        self.smoothed_history: List[np.ndarray] = []
    
    def smooth_pose_sequence(self, pose_landmarks: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to pose landmarks
        
        Args:
            pose_landmarks: Current pose landmarks array
            
        Returns:
            Smoothed pose landmarks
        """
        self.pose_history.append(pose_landmarks.copy())
        
        # Keep only recent history
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)
        
        if len(self.pose_history) == 1:
            # First frame, no smoothing
            smoothed = pose_landmarks.copy()
        else:
            # Exponential moving average
            if len(self.smoothed_history) == 0:
                smoothed = pose_landmarks.copy()
            else:
                prev_smoothed = self.smoothed_history[-1]
                smoothed = self.alpha * pose_landmarks + (1 - self.alpha) * prev_smoothed
        
        self.smoothed_history.append(smoothed.copy())
        
        # Keep smoothed history limited
        if len(self.smoothed_history) > self.window_size:
            self.smoothed_history.pop(0)
        
        return smoothed
    
    def reset(self):
        """Reset the smoother state"""
        self.pose_history.clear()
        self.smoothed_history.clear()


class BodyLanguageAnalyzer:
    """Main class combining posture classification and gesture recognition"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize body language analyzer
        
        Args:
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Initialize models
        self.posture_classifier = PostureClassifier().to(self.device)
        self.gesture_recognizer = GestureRecognizer().to(self.device)
        self.temporal_smoother = TemporalSmoother()
        
        # Load pretrained weights if available
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained model weights if available"""
        try:
            # In a real implementation, these would be loaded from MLflow or file system
            # For now, we'll initialize with random weights
            pass
        except Exception as e:
            print(f"No pretrained weights found, using random initialization: {e}")
    
    def analyze_pose_frame(self, pose_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Analyze a single frame of pose landmarks
        
        Args:
            pose_landmarks: Pose landmarks array of shape (33, 3) or flattened (99,)
            
        Returns:
            Dictionary with posture analysis results
        """
        # Ensure correct shape
        if pose_landmarks.shape == (33, 3):
            pose_landmarks = pose_landmarks.flatten()
        elif pose_landmarks.shape != (99,):
            raise ValueError(f"Expected pose landmarks shape (33, 3) or (99,), got {pose_landmarks.shape}")
        
        # Apply temporal smoothing
        smoothed_landmarks = self.temporal_smoother.smooth_pose_sequence(pose_landmarks)
        
        # Convert to tensor
        pose_tensor = torch.FloatTensor(smoothed_landmarks).unsqueeze(0).to(self.device)
        
        # Posture classification
        posture_results = self.posture_classifier.predict_with_confidence(pose_tensor)
        
        return {
            'posture_score': float(posture_results['posture_score'][0]),
            'posture_class': int(posture_results['predicted_class'][0]),
            'posture_confidence': float(posture_results['confidence'][0]),
            'posture_probabilities': posture_results['probabilities'][0].cpu().numpy().tolist()
        }
    
    def analyze_pose_sequence(self, pose_sequence: np.ndarray) -> Dict[str, any]:
        """
        Analyze a sequence of pose landmarks for gesture recognition
        
        Args:
            pose_sequence: Pose sequence array of shape (sequence_length, 33, 3) or (sequence_length, 99)
            
        Returns:
            Dictionary with gesture analysis results
        """
        # Ensure correct shape
        if len(pose_sequence.shape) == 3 and pose_sequence.shape[1:] == (33, 3):
            pose_sequence = pose_sequence.reshape(pose_sequence.shape[0], -1)
        elif len(pose_sequence.shape) != 2 or pose_sequence.shape[1] != 99:
            raise ValueError(f"Expected pose sequence shape (seq_len, 33, 3) or (seq_len, 99), got {pose_sequence.shape}")
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(pose_sequence).unsqueeze(0).to(self.device)
        
        # Gesture recognition
        gesture_results = self.gesture_recognizer.predict_gestures(sequence_tensor)
        
        return {
            'gesture_name': gesture_results['gesture_names'][0],
            'gesture_confidence': float(torch.max(gesture_results['gesture_probabilities'][0])),
            'movement_intensity': float(gesture_results['movement_intensity'][0]),
            'gesture_probabilities': gesture_results['gesture_probabilities'][0].cpu().numpy().tolist()
        }
    
    def get_comprehensive_analysis(self, pose_landmarks: np.ndarray, pose_sequence: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Get comprehensive body language analysis
        
        Args:
            pose_landmarks: Current frame pose landmarks
            pose_sequence: Optional pose sequence for gesture analysis
            
        Returns:
            Comprehensive analysis results
        """
        results = {}
        
        # Single frame posture analysis
        posture_analysis = self.analyze_pose_frame(pose_landmarks)
        results.update(posture_analysis)
        
        # Sequence-based gesture analysis if available
        if pose_sequence is not None:
            gesture_analysis = self.analyze_pose_sequence(pose_sequence)
            results.update(gesture_analysis)
        
        # Overall body language score (weighted combination)
        overall_score = (
            0.6 * posture_analysis['posture_score'] +
            0.4 * (gesture_analysis.get('gesture_confidence', 0.5) if pose_sequence is not None else 0.5)
        )
        results['overall_body_language_score'] = overall_score
        
        return results