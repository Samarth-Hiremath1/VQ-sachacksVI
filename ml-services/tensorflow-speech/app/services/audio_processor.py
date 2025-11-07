import librosa
import numpy as np
import noisereduce as nr
from scipy import signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio preprocessing pipeline for speech analysis."""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
    
    def load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio, sr)
            
            return processed_audio, self.target_sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply preprocessing pipeline to audio signal."""
        try:
            # Resample if necessary
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = self.normalize_audio(audio)
            
            # Apply noise reduction
            audio = self.reduce_noise(audio)
            
            # Apply pre-emphasis filter
            audio = self.apply_preemphasis(audio)
            
            # Trim silence
            audio = self.trim_silence(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio  # Return original audio if preprocessing fails
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio signal."""
        try:
            # Use noisereduce library for spectral gating
            reduced_noise = nr.reduce_noise(
                y=audio,
                sr=self.target_sr,
                stationary=False,
                prop_decrease=0.8
            )
            return reduced_noise
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def apply_preemphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies."""
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.target_sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            
            # Apply delta and delta-delta features
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
            
            return features.T  # Transpose to (time, features)
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            return np.zeros((100, self.n_mfcc * 3))
    
    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """Extract various spectral features from audio."""
        try:
            features = {}
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.target_sr, hop_length=self.hop_length
            )[0]
            features['spectral_centroid'] = {
                'mean': np.mean(spectral_centroids),
                'std': np.std(spectral_centroids),
                'values': spectral_centroids
            }
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.target_sr, hop_length=self.hop_length
            )[0]
            features['spectral_rolloff'] = {
                'mean': np.mean(spectral_rolloff),
                'std': np.std(spectral_rolloff),
                'values': spectral_rolloff
            }
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.target_sr, hop_length=self.hop_length
            )[0]
            features['spectral_bandwidth'] = {
                'mean': np.mean(spectral_bandwidth),
                'std': np.std(spectral_bandwidth),
                'values': spectral_bandwidth
            }
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio, hop_length=self.hop_length
            )[0]
            features['zero_crossing_rate'] = {
                'mean': np.mean(zcr),
                'std': np.std(zcr),
                'values': zcr
            }
            
            # RMS energy
            rms = librosa.feature.rms(
                y=audio, hop_length=self.hop_length
            )[0]
            features['rms_energy'] = {
                'mean': np.mean(rms),
                'std': np.std(rms),
                'values': rms
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return {}
    
    def extract_prosodic_features(self, audio: np.ndarray) -> dict:
        """Extract prosodic features (pitch, rhythm, stress)."""
        try:
            features = {}
            
            # Fundamental frequency (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.target_sr
            )
            
            # Remove unvoiced frames
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                features['f0'] = {
                    'mean': np.nanmean(f0_voiced),
                    'std': np.nanstd(f0_voiced),
                    'min': np.nanmin(f0_voiced),
                    'max': np.nanmax(f0_voiced),
                    'range': np.nanmax(f0_voiced) - np.nanmin(f0_voiced),
                    'values': f0
                }
            else:
                features['f0'] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0,
                    'values': np.zeros_like(f0)
                }
            
            # Onset detection for rhythm analysis
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.target_sr,
                hop_length=self.hop_length
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=self.target_sr)
            
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                features['rhythm'] = {
                    'onset_rate': len(onset_times) / (len(audio) / self.target_sr),
                    'avg_interval': np.mean(onset_intervals),
                    'interval_std': np.std(onset_intervals),
                    'onset_times': onset_times
                }
            else:
                features['rhythm'] = {
                    'onset_rate': 0,
                    'avg_interval': 0,
                    'interval_std': 0,
                    'onset_times': np.array([])
                }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            return {}
    
    def segment_audio_by_voice_activity(self, audio: np.ndarray, 
                                      frame_duration: float = 0.025,
                                      aggressiveness: int = 2) -> list:
        """Segment audio based on voice activity detection."""
        try:
            # Simple energy-based VAD as fallback
            frame_length = int(frame_duration * self.target_sr)
            frames = []
            
            for i in range(0, len(audio) - frame_length, frame_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame ** 2)
                frames.append((i / self.target_sr, (i + frame_length) / self.target_sr, energy))
            
            if not frames:
                return [(0, len(audio) / self.target_sr, True)]
            
            # Threshold-based voice activity detection
            energies = [f[2] for f in frames]
            threshold = np.mean(energies) * 0.1  # Adjust threshold as needed
            
            voice_segments = []
            current_segment_start = None
            
            for start_time, end_time, energy in frames:
                is_voice = energy > threshold
                
                if is_voice and current_segment_start is None:
                    current_segment_start = start_time
                elif not is_voice and current_segment_start is not None:
                    voice_segments.append((current_segment_start, end_time))
                    current_segment_start = None
            
            # Close final segment if needed
            if current_segment_start is not None:
                voice_segments.append((current_segment_start, frames[-1][1]))
            
            return voice_segments if voice_segments else [(0, len(audio) / self.target_sr)]
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            return [(0, len(audio) / self.target_sr)]
    
    def calculate_signal_quality_metrics(self, audio: np.ndarray) -> dict:
        """Calculate various signal quality metrics."""
        try:
            metrics = {}
            
            # Signal-to-noise ratio estimation
            # Simple approach: compare energy in voiced vs unvoiced regions
            voice_segments = self.segment_audio_by_voice_activity(audio)
            
            if voice_segments:
                voice_energy = 0
                silence_energy = 0
                voice_duration = 0
                silence_duration = 0
                
                for start, end in voice_segments:
                    start_sample = int(start * self.target_sr)
                    end_sample = int(end * self.target_sr)
                    segment = audio[start_sample:end_sample]
                    
                    voice_energy += np.sum(segment ** 2)
                    voice_duration += len(segment)
                
                # Estimate silence energy from gaps between voice segments
                for i in range(len(voice_segments) - 1):
                    gap_start = int(voice_segments[i][1] * self.target_sr)
                    gap_end = int(voice_segments[i + 1][0] * self.target_sr)
                    
                    if gap_end > gap_start:
                        gap_segment = audio[gap_start:gap_end]
                        silence_energy += np.sum(gap_segment ** 2)
                        silence_duration += len(gap_segment)
                
                # Calculate SNR
                if silence_duration > 0 and voice_duration > 0:
                    voice_power = voice_energy / voice_duration
                    noise_power = silence_energy / silence_duration
                    
                    if noise_power > 0:
                        snr_db = 10 * np.log10(voice_power / noise_power)
                    else:
                        snr_db = 60  # Very high SNR if no noise detected
                else:
                    snr_db = 30  # Default reasonable SNR
                
                metrics['snr_db'] = snr_db
            else:
                metrics['snr_db'] = 30
            
            # Dynamic range
            metrics['dynamic_range_db'] = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
            
            # Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
            metrics['clipping_ratio'] = clipped_samples / len(audio)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating signal quality metrics: {e}")
            return {'snr_db': 30, 'dynamic_range_db': 20, 'clipping_ratio': 0.0}