import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging


class PoseFeatureExtractor:
    """Feature extraction pipeline from pose landmarks to model input"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Define feature groups for different aspects of body language
        self.feature_groups = {
            'posture': ['shoulder_alignment', 'hip_alignment', 'spine_angle', 'head_forward_lean'],
            'arm_position': ['left_arm_angle', 'right_arm_angle', 'arm_symmetry', 'arm_openness'],
            'body_orientation': ['torso_rotation', 'hip_rotation', 'shoulder_rotation'],
            'spatial_relationships': ['arm_to_torso_ratio', 'leg_stance_width', 'center_of_mass']
        }
    
    def extract_basic_features(self, landmarks_array: np.ndarray) -> np.ndarray:
        """
        Extract basic features from pose landmarks
        
        Args:
            landmarks_array: Pose landmarks array of shape (33, 3)
            
        Returns:
            Feature vector as numpy array
        """
        if landmarks_array.shape != (33, 3):
            raise ValueError(f"Expected landmarks shape (33, 3), got {landmarks_array.shape}")
        
        features = []
        
        try:
            # Basic landmark coordinates (normalized)
            normalized_landmarks = self._normalize_landmarks(landmarks_array)
            features.extend(normalized_landmarks.flatten())
            
            # Geometric features
            geometric_features = self._extract_geometric_features(landmarks_array)
            features.extend(geometric_features)
            
            # Angular features
            angular_features = self._extract_angular_features(landmarks_array)
            features.extend(angular_features)
            
            # Distance features
            distance_features = self._extract_distance_features(landmarks_array)
            features.extend(distance_features)
            
        except Exception as e:
            self.logger.error(f"Error extracting basic features: {e}")
            # Return zero features if extraction fails
            features = [0.0] * 150  # Approximate feature count
        
        return np.array(features, dtype=np.float32)
    
    def extract_temporal_features(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from pose sequence
        
        Args:
            landmarks_sequence: Pose sequence array of shape (seq_len, 33, 3)
            
        Returns:
            Temporal feature vector
        """
        if len(landmarks_sequence.shape) != 3 or landmarks_sequence.shape[1:] != (33, 3):
            raise ValueError(f"Expected sequence shape (seq_len, 33, 3), got {landmarks_sequence.shape}")
        
        features = []
        
        try:
            # Movement velocity features
            velocity_features = self._extract_velocity_features(landmarks_sequence)
            features.extend(velocity_features)
            
            # Acceleration features
            acceleration_features = self._extract_acceleration_features(landmarks_sequence)
            features.extend(acceleration_features)
            
            # Stability features
            stability_features = self._extract_stability_features(landmarks_sequence)
            features.extend(stability_features)
            
            # Rhythm and periodicity features
            rhythm_features = self._extract_rhythm_features(landmarks_sequence)
            features.extend(rhythm_features)
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {e}")
            # Return zero features if extraction fails
            features = [0.0] * 50  # Approximate temporal feature count
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_landmarks(self, landmarks_array: np.ndarray) -> np.ndarray:
        """Normalize landmarks for translation and scale invariance"""
        # Use hip center as reference point
        left_hip = landmarks_array[23]
        right_hip = landmarks_array[24]
        hip_center = (left_hip + right_hip) / 2
        
        # Translate to hip center
        normalized = landmarks_array - hip_center
        
        # Scale based on torso size
        left_shoulder = landmarks_array[11]
        right_shoulder = landmarks_array[12]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        if torso_size > 0:
            normalized = normalized / torso_size
        
        return normalized
    
    def _extract_geometric_features(self, landmarks_array: np.ndarray) -> List[float]:
        """Extract geometric features like angles and alignments"""
        features = []
        
        # Shoulder alignment
        left_shoulder = landmarks_array[11]
        right_shoulder = landmarks_array[12]
        shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])
        features.append(shoulder_alignment)
        
        # Hip alignment
        left_hip = landmarks_array[23]
        right_hip = landmarks_array[24]
        hip_alignment = abs(left_hip[1] - right_hip[1])
        features.append(hip_alignment)
        
        # Spine angle
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        spine_vector = shoulder_center - hip_center
        spine_angle = np.arctan2(spine_vector[0], spine_vector[1])
        features.extend([np.sin(spine_angle), np.cos(spine_angle)])
        
        # Head position
        nose = landmarks_array[0]
        head_shoulder_offset = nose - shoulder_center
        features.extend(head_shoulder_offset.tolist())
        
        return features
    
    def _extract_angular_features(self, landmarks_array: np.ndarray) -> List[float]:
        """Extract angular features for joints and body parts"""
        features = []
        
        # Arm angles
        left_shoulder = landmarks_array[11]
        left_elbow = landmarks_array[13]
        left_wrist = landmarks_array[15]
        
        right_shoulder = landmarks_array[12]
        right_elbow = landmarks_array[14]
        right_wrist = landmarks_array[16]
        
        # Left arm angle
        upper_arm_left = left_elbow - left_shoulder
        forearm_left = left_wrist - left_elbow
        left_arm_angle = self._calculate_angle(upper_arm_left, forearm_left)
        features.extend([np.sin(left_arm_angle), np.cos(left_arm_angle)])
        
        # Right arm angle
        upper_arm_right = right_elbow - right_shoulder
        forearm_right = right_wrist - right_elbow
        right_arm_angle = self._calculate_angle(upper_arm_right, forearm_right)
        features.extend([np.sin(right_arm_angle), np.cos(right_arm_angle)])
        
        # Leg angles
        left_hip = landmarks_array[23]
        left_knee = landmarks_array[25]
        left_ankle = landmarks_array[27]
        
        right_hip = landmarks_array[24]
        right_knee = landmarks_array[26]
        right_ankle = landmarks_array[28]
        
        # Left leg angle
        thigh_left = left_knee - left_hip
        shin_left = left_ankle - left_knee
        left_leg_angle = self._calculate_angle(thigh_left, shin_left)
        features.extend([np.sin(left_leg_angle), np.cos(left_leg_angle)])
        
        # Right leg angle
        thigh_right = right_knee - right_hip
        shin_right = right_ankle - right_knee
        right_leg_angle = self._calculate_angle(thigh_right, shin_right)
        features.extend([np.sin(right_leg_angle), np.cos(right_leg_angle)])
        
        return features
    
    def _extract_distance_features(self, landmarks_array: np.ndarray) -> List[float]:
        """Extract distance-based features"""
        features = []
        
        # Hand distances
        left_wrist = landmarks_array[15]
        right_wrist = landmarks_array[16]
        hand_distance = np.linalg.norm(left_wrist - right_wrist)
        features.append(hand_distance)
        
        # Foot distances
        left_ankle = landmarks_array[27]
        right_ankle = landmarks_array[28]
        foot_distance = np.linalg.norm(left_ankle - right_ankle)
        features.append(foot_distance)
        
        # Body width (shoulder distance)
        left_shoulder = landmarks_array[11]
        right_shoulder = landmarks_array[12]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        features.append(shoulder_width)
        
        # Body height (head to hip distance)
        nose = landmarks_array[0]
        left_hip = landmarks_array[23]
        right_hip = landmarks_array[24]
        hip_center = (left_hip + right_hip) / 2
        body_height = np.linalg.norm(nose - hip_center)
        features.append(body_height)
        
        return features
    
    def _extract_velocity_features(self, landmarks_sequence: np.ndarray) -> List[float]:
        """Extract velocity features from pose sequence"""
        features = []
        
        # Calculate velocities for key landmarks
        key_landmarks = [0, 11, 12, 15, 16, 23, 24]  # Nose, shoulders, wrists, hips
        
        for landmark_idx in key_landmarks:
            landmark_positions = landmarks_sequence[:, landmark_idx, :]
            velocities = np.diff(landmark_positions, axis=0)
            
            # Velocity statistics
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            features.extend([
                np.mean(velocity_magnitudes),
                np.std(velocity_magnitudes),
                np.max(velocity_magnitudes)
            ])
        
        return features
    
    def _extract_acceleration_features(self, landmarks_sequence: np.ndarray) -> List[float]:
        """Extract acceleration features from pose sequence"""
        features = []
        
        # Calculate accelerations for key landmarks
        key_landmarks = [0, 15, 16]  # Nose, wrists (most expressive)
        
        for landmark_idx in key_landmarks:
            landmark_positions = landmarks_sequence[:, landmark_idx, :]
            velocities = np.diff(landmark_positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            # Acceleration statistics
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            if len(acceleration_magnitudes) > 0:
                features.extend([
                    np.mean(acceleration_magnitudes),
                    np.std(acceleration_magnitudes)
                ])
            else:
                features.extend([0.0, 0.0])
        
        return features
    
    def _extract_stability_features(self, landmarks_sequence: np.ndarray) -> List[float]:
        """Extract stability and steadiness features"""
        features = []
        
        # Center of mass stability
        center_of_mass = np.mean(landmarks_sequence, axis=1)  # Average across landmarks
        com_movement = np.diff(center_of_mass, axis=0)
        com_stability = np.std(np.linalg.norm(com_movement, axis=1))
        features.append(com_stability)
        
        # Head stability (important for presentation confidence)
        head_positions = landmarks_sequence[:, 0, :]  # Nose landmark
        head_movement = np.diff(head_positions, axis=0)
        head_stability = np.std(np.linalg.norm(head_movement, axis=1))
        features.append(head_stability)
        
        # Torso stability
        torso_landmarks = [11, 12, 23, 24]  # Shoulders and hips
        torso_positions = np.mean(landmarks_sequence[:, torso_landmarks, :], axis=1)
        torso_movement = np.diff(torso_positions, axis=0)
        torso_stability = np.std(np.linalg.norm(torso_movement, axis=1))
        features.append(torso_stability)
        
        return features
    
    def _extract_rhythm_features(self, landmarks_sequence: np.ndarray) -> List[float]:
        """Extract rhythm and periodicity features"""
        features = []
        
        # Hand movement rhythm (for gestures)
        left_wrist = landmarks_sequence[:, 15, :]
        right_wrist = landmarks_sequence[:, 16, :]
        
        # Calculate movement energy over time
        left_movement = np.linalg.norm(np.diff(left_wrist, axis=0), axis=1)
        right_movement = np.linalg.norm(np.diff(right_wrist, axis=0), axis=1)
        
        # Movement rhythm features
        total_movement = left_movement + right_movement
        if len(total_movement) > 0:
            movement_variance = np.var(total_movement)
            movement_peaks = len([i for i in range(1, len(total_movement)-1) 
                                if total_movement[i] > total_movement[i-1] and 
                                   total_movement[i] > total_movement[i+1]])
            features.extend([movement_variance, movement_peaks / len(total_movement)])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _calculate_angle(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def fit_scaler(self, feature_arrays: List[np.ndarray]):
        """Fit the feature scaler on training data"""
        all_features = np.vstack(feature_arrays)
        self.scaler.fit(all_features)
        self.is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.is_fitted:
            self.logger.warning("Scaler not fitted, returning raw features")
            return features
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return self.scaler.transform(features)
    
    def prepare_model_input(self, landmarks_array: np.ndarray, 
                          landmarks_sequence: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Prepare complete feature vector for model input
        
        Args:
            landmarks_array: Single frame landmarks
            landmarks_sequence: Optional sequence for temporal features
            
        Returns:
            PyTorch tensor ready for model input
        """
        # Extract basic features
        basic_features = self.extract_basic_features(landmarks_array)
        
        # Extract temporal features if sequence is provided
        if landmarks_sequence is not None:
            temporal_features = self.extract_temporal_features(landmarks_sequence)
            # Combine basic and temporal features
            all_features = np.concatenate([basic_features, temporal_features])
        else:
            all_features = basic_features
        
        # Transform features
        transformed_features = self.transform_features(all_features)
        
        # Convert to PyTorch tensor
        return torch.FloatTensor(transformed_features)