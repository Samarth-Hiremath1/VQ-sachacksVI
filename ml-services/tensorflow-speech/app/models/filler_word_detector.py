import re
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FillerWordAnalysis:
    total_filler_words: int
    filler_word_rate: float  # fillers per minute
    filler_word_types: Dict[str, int]
    filler_word_positions: List[Tuple[str, float]]  # (word, timestamp)
    confidence_scores: List[float]

class FillerWordDetector:
    """Hybrid filler word detection using rule-based and ML approaches."""
    
    def __init__(self):
        # Common filler words and patterns
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well',
            'actually', 'basically', 'literally', 'right', 'okay', 'ok',
            'i mean', 'sort of', 'kind of', 'you see', 'obviously'
        }
        
        # Extended patterns for detection
        self.filler_patterns = [
            r'\b(um+|uh+|er+|ah+)\b',
            r'\b(like)\b(?!\s+(this|that|it|he|she|they|we|you|i))',
            r'\byou\s+know\b',
            r'\bi\s+mean\b',
            r'\bsort\s+of\b',
            r'\bkind\s+of\b',
            r'\byou\s+see\b',
            r'\b(so|well|actually|basically|literally|obviously)\b(?=\s)',
            r'\b(right|okay|ok)\b(?=[\s,.])',
        ]
        
        # Build ML model for filler word classification
        self.ml_model = self._build_filler_detection_model()
    
    def _build_filler_detection_model(self) -> tf.keras.Model:
        """Build TensorFlow model for filler word detection enhancement."""
        # Simple model for context-aware filler detection
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50,)),  # Context features
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid', name='is_filler')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def detect_filler_words_rule_based(self, transcript: str, timestamps: List[float] = None) -> FillerWordAnalysis:
        """Rule-based filler word detection."""
        if not transcript:
            return FillerWordAnalysis(0, 0.0, {}, [], [])
        
        transcript_lower = transcript.lower()
        filler_positions = []
        filler_types = {}
        
        # Split into words with positions
        words = transcript.split()
        word_positions = []
        
        # Calculate approximate word timestamps
        if timestamps and len(timestamps) >= 2:
            duration = timestamps[-1] - timestamps[0]
            words_per_second = len(words) / duration if duration > 0 else 1
            
            for i, word in enumerate(words):
                timestamp = timestamps[0] + (i / words_per_second) if timestamps else i
                word_positions.append((word.lower(), timestamp))
        else:
            word_positions = [(word.lower(), i) for i, word in enumerate(words)]
        
        # Check each word against filler patterns
        for word, timestamp in word_positions:
            # Direct word matching
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.filler_words:
                filler_positions.append((clean_word, timestamp))
                filler_types[clean_word] = filler_types.get(clean_word, 0) + 1
        
        # Check multi-word patterns
        for pattern in self.filler_patterns:
            matches = re.finditer(pattern, transcript_lower)
            for match in matches:
                filler_text = match.group().strip()
                # Estimate timestamp for multi-word fillers
                word_index = len(transcript_lower[:match.start()].split())
                timestamp = word_positions[min(word_index, len(word_positions) - 1)][1] if word_positions else 0
                
                filler_positions.append((filler_text, timestamp))
                filler_types[filler_text] = filler_types.get(filler_text, 0) + 1
        
        # Calculate rate (fillers per minute)
        duration_minutes = (timestamps[-1] - timestamps[0]) / 60 if timestamps and len(timestamps) >= 2 else len(words) / 150
        filler_rate = len(filler_positions) / duration_minutes if duration_minutes > 0 else 0
        
        return FillerWordAnalysis(
            total_filler_words=len(filler_positions),
            filler_word_rate=filler_rate,
            filler_word_types=filler_types,
            filler_word_positions=filler_positions,
            confidence_scores=[1.0] * len(filler_positions)  # Rule-based has high confidence
        )
    
    def extract_context_features(self, words: List[str], target_index: int, window_size: int = 5) -> np.ndarray:
        """Extract context features for ML-based filler detection."""
        features = np.zeros(50)  # Fixed feature size
        
        try:
            # Get context window
            start_idx = max(0, target_index - window_size)
            end_idx = min(len(words), target_index + window_size + 1)
            context_words = words[start_idx:end_idx]
            
            # Basic features
            target_word = words[target_index].lower() if target_index < len(words) else ""
            
            # Word length features
            features[0] = len(target_word)
            features[1] = len(target_word.split())
            
            # Position features
            features[2] = target_index / len(words) if len(words) > 0 else 0
            features[3] = 1 if target_index == 0 else 0  # Start of sentence
            features[4] = 1 if target_index == len(words) - 1 else 0  # End of sentence
            
            # Context features
            for i, word in enumerate(context_words[:10]):  # Limit to 10 context words
                # Simple word embedding simulation (character-based)
                word_hash = hash(word.lower()) % 1000
                features[5 + i] = word_hash / 1000.0
            
            # Repetition features
            word_count = context_words.count(target_word)
            features[15] = min(word_count / len(context_words), 1.0) if context_words else 0
            
            # Punctuation features
            features[16] = 1 if any(p in target_word for p in '.,!?;:') else 0
            
            # Length statistics
            if context_words:
                avg_length = np.mean([len(w) for w in context_words])
                features[17] = len(target_word) / avg_length if avg_length > 0 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
            return features
    
    def detect_filler_words_ml_enhanced(self, transcript: str, timestamps: List[float] = None) -> FillerWordAnalysis:
        """ML-enhanced filler word detection."""
        # Start with rule-based detection
        rule_based_result = self.detect_filler_words_rule_based(transcript, timestamps)
        
        if not transcript:
            return rule_based_result
        
        words = transcript.split()
        enhanced_positions = []
        enhanced_types = {}
        confidence_scores = []
        
        # Use ML model to validate and enhance rule-based results
        for i, word in enumerate(words):
            context_features = self.extract_context_features(words, i)
            
            # Get ML prediction
            try:
                ml_prediction = self.ml_model.predict(
                    np.expand_dims(context_features, axis=0), 
                    verbose=0
                )[0][0]
                
                # Combine rule-based and ML predictions
                word_lower = word.lower().strip('.,!?;:')
                is_rule_based_filler = any(word_lower == pos[0] for pos in rule_based_result.filler_word_positions)
                
                # Decision logic: high ML confidence OR rule-based detection
                if ml_prediction > 0.7 or (is_rule_based_filler and ml_prediction > 0.3):
                    timestamp = timestamps[i] if timestamps and i < len(timestamps) else i
                    enhanced_positions.append((word_lower, timestamp))
                    enhanced_types[word_lower] = enhanced_types.get(word_lower, 0) + 1
                    confidence_scores.append(float(ml_prediction))
                    
            except Exception as e:
                logger.error(f"Error in ML prediction for word '{word}': {e}")
                # Fall back to rule-based result for this word
                if any(word.lower() == pos[0] for pos in rule_based_result.filler_word_positions):
                    timestamp = timestamps[i] if timestamps and i < len(timestamps) else i
                    enhanced_positions.append((word.lower(), timestamp))
                    enhanced_types[word.lower()] = enhanced_types.get(word.lower(), 0) + 1
                    confidence_scores.append(0.8)  # Default confidence for rule-based
        
        # Calculate enhanced rate
        duration_minutes = (timestamps[-1] - timestamps[0]) / 60 if timestamps and len(timestamps) >= 2 else len(words) / 150
        filler_rate = len(enhanced_positions) / duration_minutes if duration_minutes > 0 else 0
        
        return FillerWordAnalysis(
            total_filler_words=len(enhanced_positions),
            filler_word_rate=filler_rate,
            filler_word_types=enhanced_types,
            filler_word_positions=enhanced_positions,
            confidence_scores=confidence_scores
        )
    
    def get_filler_word_recommendations(self, analysis: FillerWordAnalysis) -> List[str]:
        """Generate recommendations based on filler word analysis."""
        recommendations = []
        
        if analysis.filler_word_rate > 10:  # More than 10 fillers per minute
            recommendations.append("Try to reduce filler words by practicing with pauses instead")
            recommendations.append("Record yourself and identify your most common filler words")
        elif analysis.filler_word_rate > 5:
            recommendations.append("Consider being more mindful of filler word usage")
            recommendations.append("Practice speaking more slowly to reduce fillers")
        else:
            recommendations.append("Good job keeping filler words to a minimum!")
        
        # Specific recommendations based on filler types
        if 'like' in analysis.filler_word_types and analysis.filler_word_types['like'] > 3:
            recommendations.append("Try to replace 'like' with more specific descriptive words")
        
        if any(word in analysis.filler_word_types for word in ['um', 'uh', 'er']):
            recommendations.append("Practice pausing silently instead of using vocal fillers")
        
        if 'you know' in analysis.filler_word_types:
            recommendations.append("Ensure your audience understands by asking direct questions instead of 'you know'")
        
        return recommendations
    
    def train_ml_model(self, training_data: List[Tuple[str, List[bool]]], epochs: int = 10):
        """Train the ML model with labeled data."""
        X, y = [], []
        
        for transcript, labels in training_data:
            words = transcript.split()
            for i, is_filler in enumerate(labels):
                if i < len(words):
                    features = self.extract_context_features(words, i)
                    X.append(features)
                    y.append(1 if is_filler else 0)
        
        if X and y:
            X = np.array(X)
            y = np.array(y)
            
            self.ml_model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
    
    def save_model(self, path: str):
        """Save the trained ML model."""
        self.ml_model.save(f"{path}/filler_detection_model")
    
    def load_model(self, path: str):
        """Load a trained ML model."""
        try:
            self.ml_model = tf.keras.models.load_model(f"{path}/filler_detection_model")
        except Exception as e:
            logger.warning(f"Could not load filler detection model from {path}: {e}")
            logger.info("Using default untrained model")