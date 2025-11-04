import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass


@dataclass
class PoseLandmark:
    """Data class for pose landmark with coordinates and visibility"""
    x: float
    y: float
    z: float
    visibility: float


class MediaPipePoseDetector:
    """MediaPipe pose detection integration for landmark extraction"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose detector
        
        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Complexity of pose model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to generate segmentation mask
            smooth_segmentation: Whether to smooth segmentation mask
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Landmark indices for specific body parts
        self.landmark_indices = {
            'face': list(range(0, 11)),
            'torso': [11, 12, 23, 24],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }   
 def detect_pose(self, image: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Detect pose landmarks in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing pose landmarks and metadata, or None if no pose detected
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks is None:
                return None
            
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append(PoseLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                ))
            
            # Convert to numpy array for ML processing
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            
            # Calculate additional metrics
            pose_metrics = self._calculate_pose_metrics(landmarks_array)
            
            return {
                'landmarks': landmarks,
                'landmarks_array': landmarks_array,
                'pose_metrics': pose_metrics,
                'segmentation_mask': results.segmentation_mask,
                'detection_confidence': self._calculate_detection_confidence(landmarks)
            }
            
        except Exception as e:
            self.logger.error(f"Error in pose detection: {e}")
            return None
    
    def detect_pose_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Detect pose landmarks in a video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None for all frames)
            
        Returns:
            List of pose detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        pose_results = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                pose_result = self.detect_pose(frame)
                if pose_result is not None:
                    pose_result['frame_number'] = frame_count
                    pose_result['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                    pose_results.append(pose_result)
                
                frame_count += 1
                
        finally:
            cap.release()
        
        return pose_results   
 def _calculate_pose_metrics(self, landmarks_array: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional pose metrics from landmarks
        
        Args:
            landmarks_array: Pose landmarks as numpy array (33, 3)
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            # Shoulder alignment (difference in y-coordinates)
            left_shoulder = landmarks_array[11]  # Left shoulder
            right_shoulder = landmarks_array[12]  # Right shoulder
            metrics['shoulder_alignment'] = abs(left_shoulder[1] - right_shoulder[1])
            
            # Hip alignment
            left_hip = landmarks_array[23]  # Left hip
            right_hip = landmarks_array[24]  # Right hip
            metrics['hip_alignment'] = abs(left_hip[1] - right_hip[1])
            
            # Spine straightness (angle between shoulders and hips)
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            spine_vector = shoulder_center - hip_center
            metrics['spine_angle'] = np.arctan2(spine_vector[0], spine_vector[1]) * 180 / np.pi
            
            # Head position relative to shoulders
            nose = landmarks_array[0]  # Nose
            head_shoulder_offset = nose[0] - shoulder_center[0]
            metrics['head_forward_lean'] = abs(head_shoulder_offset)
            
            # Arm positions
            left_elbow = landmarks_array[13]
            right_elbow = landmarks_array[14]
            left_wrist = landmarks_array[15]
            right_wrist = landmarks_array[16]
            
            # Calculate arm angles
            left_arm_vector = left_wrist - left_elbow
            right_arm_vector = right_wrist - right_elbow
            
            metrics['left_arm_angle'] = np.arctan2(left_arm_vector[1], left_arm_vector[0]) * 180 / np.pi
            metrics['right_arm_angle'] = np.arctan2(right_arm_vector[1], right_arm_vector[0]) * 180 / np.pi
            
            # Overall posture score (0-1, higher is better)
            posture_score = 1.0
            posture_score -= min(metrics['shoulder_alignment'] * 2, 0.3)  # Penalize shoulder misalignment
            posture_score -= min(metrics['hip_alignment'] * 2, 0.3)      # Penalize hip misalignment
            posture_score -= min(abs(metrics['spine_angle']) / 45, 0.2)  # Penalize spine lean
            posture_score -= min(metrics['head_forward_lean'] * 3, 0.2)  # Penalize forward head
            
            metrics['posture_score'] = max(0.0, posture_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating pose metrics: {e}")
            # Return default metrics if calculation fails
            metrics = {
                'shoulder_alignment': 0.0,
                'hip_alignment': 0.0,
                'spine_angle': 0.0,
                'head_forward_lean': 0.0,
                'left_arm_angle': 0.0,
                'right_arm_angle': 0.0,
                'posture_score': 0.5
            }
        
        return metrics
    
    def _calculate_detection_confidence(self, landmarks: List[PoseLandmark]) -> float:
        """
        Calculate overall detection confidence based on landmark visibility
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Overall confidence score (0-1)
        """
        if not landmarks:
            return 0.0
        
        # Calculate average visibility of key landmarks
        key_landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Nose, shoulders, elbows, wrists, hips
        key_landmarks = [landmarks[i] for i in key_landmark_indices if i < len(landmarks)]
        
        if not key_landmarks:
            return 0.0
        
        avg_visibility = sum(lm.visibility for lm in key_landmarks) / len(key_landmarks)
        return avg_visibility 
   def normalize_landmarks(self, landmarks_array: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be translation and scale invariant
        
        Args:
            landmarks_array: Raw pose landmarks array
            
        Returns:
            Normalized landmarks array
        """
        # Use hip center as reference point
        left_hip = landmarks_array[23]
        right_hip = landmarks_array[24]
        hip_center = (left_hip + right_hip) / 2
        
        # Translate to hip center
        normalized = landmarks_array - hip_center
        
        # Scale based on torso size (shoulder to hip distance)
        left_shoulder = landmarks_array[11]
        right_shoulder = landmarks_array[12]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        if torso_size > 0:
            normalized = normalized / torso_size
        
        return normalized
    
    def close(self):
        """Close the pose detector and release resources"""
        if hasattr(self, 'pose'):
            self.pose.close()


class PoseSequenceProcessor:
    """Process sequences of pose landmarks for temporal analysis"""
    
    def __init__(self, sequence_length: int = 30, overlap: float = 0.5):
        """
        Initialize pose sequence processor
        
        Args:
            sequence_length: Length of pose sequences to extract
            overlap: Overlap between consecutive sequences (0-1)
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.step_size = int(sequence_length * (1 - overlap))
    
    def extract_sequences(self, pose_results: List[Dict[str, any]]) -> List[np.ndarray]:
        """
        Extract overlapping sequences from pose detection results
        
        Args:
            pose_results: List of pose detection results
            
        Returns:
            List of pose sequences as numpy arrays
        """
        if len(pose_results) < self.sequence_length:
            return []
        
        sequences = []
        landmarks_list = []
        
        # Extract landmarks arrays
        for result in pose_results:
            if 'landmarks_array' in result:
                landmarks_list.append(result['landmarks_array'])
        
        # Create overlapping sequences
        for i in range(0, len(landmarks_list) - self.sequence_length + 1, self.step_size):
            sequence = np.array(landmarks_list[i:i + self.sequence_length])
            sequences.append(sequence)
        
        return sequences