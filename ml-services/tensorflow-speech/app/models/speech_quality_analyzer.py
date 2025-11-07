import tensorflow as tf
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SpeechQualityMetrics:
    clarity_score: float
    volume_variation_score: float
    pace_score: float
    overall_quality: float
    speaking_rate_wpm: float
    volume_consistency: float
    pitch_variation: float

class SpeechQualityAnalyzer:
    """TensorFlow-based speech quality analyzer for presentation coaching."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
        # Initialize models
        self.clarity_model = self._build_clarity_model()
        self.volume_model = self._build_volume_model()
        self.pace_model = self._build_pace_model()
        
        if model_path:
            self.load_models(model_path)
    
    def _build_clarity_model(self) -> tf.keras.Model:
        """Build TensorFlow model for speech clarity assessment."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.n_mfcc)),
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid', name='clarity_score')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_volume_model(self) -> tf.keras.Model:
        """Build TensorFlow model for volume variation analysis."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 1)),  # RMS energy over time
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='volume_consistency')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_pace_model(self) -> tf.keras.Model:
        """Build TensorFlow model for speaking pace analysis."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),  # Pace features
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='pace_score')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio signal."""
        try:
            # Ensure audio is the right sample rate
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Transpose to (time, features) format
            return mfcc.T
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            # Return zeros if extraction fails
            return np.zeros((100, self.n_mfcc))
    
    def extract_volume_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract volume/energy features from audio signal."""
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(
                y=audio,
                hop_length=self.hop_length
            )[0]
            
            return rms.reshape(-1, 1)
            
        except Exception as e:
            logger.error(f"Error extracting volume features: {e}")
            return np.zeros((100, 1))
    
    def extract_pace_features(self, audio: np.ndarray, transcript: str = "") -> np.ndarray:
        """Extract speaking pace features."""
        try:
            duration = len(audio) / self.sample_rate
            
            # Basic pace metrics
            word_count = len(transcript.split()) if transcript else 0
            speaking_rate = word_count / duration * 60 if duration > 0 else 0
            
            # Audio-based pace features
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            
            # Calculate pace statistics
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                avg_interval = np.mean(onset_intervals)
                interval_std = np.std(onset_intervals)
                interval_cv = interval_std / avg_interval if avg_interval > 0 else 0
            else:
                avg_interval = interval_std = interval_cv = 0
            
            # Spectral features for pace
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            features = np.array([
                speaking_rate,
                avg_interval,
                interval_std,
                interval_cv,
                len(onset_times) / duration if duration > 0 else 0,
                spectral_centroid,
                spectral_rolloff,
                zero_crossing_rate,
                duration,
                word_count / duration if duration > 0 else 0
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pace features: {e}")
            return np.zeros(10)
    
    def analyze_speech_quality(self, audio: np.ndarray, transcript: str = "") -> SpeechQualityMetrics:
        """Analyze speech quality using TensorFlow models."""
        try:
            # Extract features
            mfcc_features = self.extract_mfcc_features(audio)
            volume_features = self.extract_volume_features(audio)
            pace_features = self.extract_pace_features(audio, transcript)
            
            # Prepare inputs for models
            mfcc_input = np.expand_dims(mfcc_features, axis=0)
            volume_input = np.expand_dims(volume_features, axis=0)
            pace_input = np.expand_dims(pace_features, axis=0)
            
            # Get predictions
            clarity_score = float(self.clarity_model.predict(mfcc_input, verbose=0)[0][0])
            volume_consistency = float(self.volume_model.predict(volume_input, verbose=0)[0][0])
            pace_score = float(self.pace_model.predict(pace_input, verbose=0)[0][0])
            
            # Calculate additional metrics
            speaking_rate_wpm = pace_features[0]
            
            # Volume variation analysis
            volume_variation_score = self._calculate_volume_variation(volume_features.flatten())
            
            # Pitch variation analysis
            pitch_variation = self._calculate_pitch_variation(audio)
            
            # Calculate overall quality score
            overall_quality = (clarity_score * 0.4 + 
                             volume_consistency * 0.3 + 
                             pace_score * 0.3)
            
            return SpeechQualityMetrics(
                clarity_score=clarity_score,
                volume_variation_score=volume_variation_score,
                pace_score=pace_score,
                overall_quality=overall_quality,
                speaking_rate_wpm=speaking_rate_wpm,
                volume_consistency=volume_consistency,
                pitch_variation=pitch_variation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing speech quality: {e}")
            # Return default metrics on error
            return SpeechQualityMetrics(
                clarity_score=0.5,
                volume_variation_score=0.5,
                pace_score=0.5,
                overall_quality=0.5,
                speaking_rate_wpm=120.0,
                volume_consistency=0.5,
                pitch_variation=0.5
            )
    
    def _calculate_volume_variation(self, rms_values: np.ndarray) -> float:
        """Calculate volume variation score (0-1, higher is better consistency)."""
        if len(rms_values) == 0:
            return 0.5
        
        # Calculate coefficient of variation
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)
        
        if mean_rms == 0:
            return 0.5
        
        cv = std_rms / mean_rms
        
        # Convert to score (lower variation = higher score)
        # Normalize CV to 0-1 range (assuming CV > 1 is poor)
        score = max(0, 1 - cv)
        return min(1, score)
    
    def _calculate_pitch_variation(self, audio: np.ndarray) -> float:
        """Calculate pitch variation score."""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                threshold=0.1
            )
            
            # Get fundamental frequency
            f0 = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0.append(pitch)
            
            if len(f0) < 2:
                return 0.5
            
            f0 = np.array(f0)
            
            # Calculate pitch variation metrics
            pitch_std = np.std(f0)
            pitch_mean = np.mean(f0)
            
            if pitch_mean == 0:
                return 0.5
            
            # Normalize pitch variation (good speakers have moderate variation)
            cv = pitch_std / pitch_mean
            
            # Optimal range is around 0.1-0.3 CV
            if 0.1 <= cv <= 0.3:
                score = 1.0
            elif cv < 0.1:
                score = cv / 0.1  # Too monotone
            else:
                score = max(0, 1 - (cv - 0.3) / 0.7)  # Too variable
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating pitch variation: {e}")
            return 0.5
    
    def get_pace_recommendations(self, speaking_rate_wpm: float) -> List[str]:
        """Generate speaking pace recommendations."""
        recommendations = []
        
        if speaking_rate_wpm < 120:
            recommendations.append("Consider speaking slightly faster to maintain audience engagement")
            recommendations.append("Practice with a metronome to increase your speaking pace")
        elif speaking_rate_wpm > 180:
            recommendations.append("Try to slow down your speaking pace for better clarity")
            recommendations.append("Practice pausing between key points")
        else:
            recommendations.append("Your speaking pace is in the optimal range")
        
        return recommendations
    
    def save_models(self, path: str):
        """Save trained models to disk."""
        self.clarity_model.save(f"{path}/clarity_model")
        self.volume_model.save(f"{path}/volume_model")
        self.pace_model.save(f"{path}/pace_model")
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        try:
            self.clarity_model = tf.keras.models.load_model(f"{path}/clarity_model")
            self.volume_model = tf.keras.models.load_model(f"{path}/volume_model")
            self.pace_model = tf.keras.models.load_model(f"{path}/pace_model")
        except Exception as e:
            logger.warning(f"Could not load models from {path}: {e}")
            logger.info("Using default untrained models")