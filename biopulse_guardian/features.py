#!/usr/bin/env python3
"""
BioPulse Guardian - Feature Extraction Module

This module handles feature extraction from raw biometric data for analysis
and anomaly detection. It provides various statistical and time-based features.

Author: BioPulse Guardian Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.fft import fft
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiometricFeatureExtractor:
    """
    Extracts various features from biometric time series data
    for use in anomaly detection and health analysis.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_config = {
            'window_size': 60,  # 60 seconds window
            'overlap': 0.5,     # 50% overlap
            'heart_rate_bands': {
                'resting': (60, 100),
                'moderate': (100, 150),
                'high': (150, 200)
            }
        }
        
    def extract_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from time series data.
        
        Args:
            data: 1D numpy array of biometric values
            
        Returns:
            Dictionary containing statistical features
        """
        if len(data) == 0:
            return {}
            
        try:
            features = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'rms': float(np.sqrt(np.mean(data**2))),
                'variance': float(np.var(data))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting statistical features: {e}")
            return {}
    
    def extract_time_domain_features(self, data: np.ndarray, 
                                   timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract time-domain features from biometric data.
        
        Args:
            data: 1D numpy array of biometric values
            timestamps: Optional timestamps for the data
            
        Returns:
            Dictionary containing time-domain features
        """
        if len(data) < 2:
            return {}
            
        try:
            features = {}
            
            # Rate of change features
            diff = np.diff(data)
            features.update({
                'mean_diff': float(np.mean(diff)),
                'std_diff': float(np.std(diff)),
                'max_increase': float(np.max(diff)),
                'max_decrease': float(np.min(diff)),
                'zero_crossings': int(np.sum(np.diff(np.sign(diff)) != 0))
            })
            
            # Trend analysis
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            features.update({
                'trend_slope': float(slope),
                'trend_r_squared': float(r_value**2),
                'trend_p_value': float(p_value)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting time domain features: {e}")
            return {}
    
    def extract_frequency_features(self, data: np.ndarray, 
                                 sample_rate: float = 1.0) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Args:
            data: 1D numpy array of biometric values
            sample_rate: Sampling rate of the data
            
        Returns:
            Dictionary containing frequency-domain features
        """
        if len(data) < 4:
            return {}
            
        try:
            # Apply FFT
            fft_values = np.abs(fft(data))
            freqs = np.fft.fftfreq(len(data), 1/sample_rate)
            
            # Only use positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = fft_values[:len(fft_values)//2]
            
            features = {
                'dominant_freq': float(positive_freqs[np.argmax(positive_fft)]),
                'spectral_centroid': float(np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)),
                'spectral_rolloff': self._calculate_spectral_rolloff(positive_freqs, positive_fft),
                'spectral_bandwidth': self._calculate_spectral_bandwidth(positive_freqs, positive_fft)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting frequency features: {e}")
            return {}
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, magnitudes: np.ndarray, 
                                   rolloff_point: float = 0.85) -> float:
        """
        Calculate spectral rolloff frequency.
        
        Args:
            freqs: Frequency array
            magnitudes: Magnitude array
            rolloff_point: Rolloff threshold (default: 85%)
            
        Returns:
            Spectral rolloff frequency
        """
        total_energy = np.sum(magnitudes**2)
        cumulative_energy = np.cumsum(magnitudes**2)
        rolloff_idx = np.where(cumulative_energy >= rolloff_point * total_energy)[0]
        
        if len(rolloff_idx) > 0:
            return float(freqs[rolloff_idx[0]])
        else:
            return float(freqs[-1])
    
    def _calculate_spectral_bandwidth(self, freqs: np.ndarray, magnitudes: np.ndarray) -> float:
        """
        Calculate spectral bandwidth.
        
        Args:
            freqs: Frequency array
            magnitudes: Magnitude array
            
        Returns:
            Spectral bandwidth
        """
        centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes))
        return float(bandwidth)
    
    def extract_heart_rate_features(self, heart_rate_data: np.ndarray) -> Dict[str, float]:
        """
        Extract heart rate specific features.
        
        Args:
            heart_rate_data: Array of heart rate values
            
        Returns:
            Dictionary containing heart rate features
        """
        if len(heart_rate_data) == 0:
            return {}
            
        try:
            features = {}
            bands = self.feature_config['heart_rate_bands']
            
            # Zone analysis
            for zone, (low, high) in bands.items():
                in_zone = np.logical_and(heart_rate_data >= low, heart_rate_data <= high)
                features[f'time_in_{zone}_zone'] = float(np.sum(in_zone) / len(heart_rate_data))
            
            # Heart rate variability approximation
            if len(heart_rate_data) > 1:
                rr_intervals = 60.0 / heart_rate_data  # Approximate RR intervals
                features.update({
                    'hrv_rmssd': float(np.sqrt(np.mean(np.diff(rr_intervals)**2))),
                    'hrv_sdnn': float(np.std(rr_intervals))
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting heart rate features: {e}")
            return {}
    
    def extract_comprehensive_features(self, data: pd.DataFrame, 
                                     timestamp_col: str = 'timestamp',
                                     value_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract comprehensive features from biometric DataFrame.
        
        Args:
            data: DataFrame containing biometric data
            timestamp_col: Name of timestamp column
            value_cols: List of columns to extract features from
            
        Returns:
            Dictionary containing all extracted features
        """
        if data.empty:
            return {}
            
        try:
            all_features = {}
            
            if value_cols is None:
                value_cols = [col for col in data.columns if col != timestamp_col]
            
            for col in value_cols:
                if col not in data.columns:
                    continue
                    
                values = data[col].dropna().values
                if len(values) == 0:
                    continue
                
                # Extract different types of features
                stat_features = self.extract_statistical_features(values)
                time_features = self.extract_time_domain_features(values)
                freq_features = self.extract_frequency_features(values)
                
                # Add column prefix
                for feature_dict in [stat_features, time_features, freq_features]:
                    for key, value in feature_dict.items():
                        all_features[f'{col}_{key}'] = value
                
                # Special handling for heart rate
                if 'heart_rate' in col.lower() or 'hr' in col.lower():
                    hr_features = self.extract_heart_rate_features(values)
                    for key, value in hr_features.items():
                        all_features[f'{col}_{key}'] = value
            
            # Add metadata
            all_features.update({
                'n_records': len(data),
                'time_span_minutes': self._calculate_time_span(data, timestamp_col),
                'extraction_timestamp': datetime.now().isoformat()
            })
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive features: {e}")
            return {}
    
    def _calculate_time_span(self, data: pd.DataFrame, timestamp_col: str) -> float:
        """
        Calculate time span of data in minutes.
        
        Args:
            data: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            Time span in minutes
        """
        try:
            if timestamp_col not in data.columns or len(data) < 2:
                return 0.0
                
            timestamps = pd.to_datetime(data[timestamp_col])
            time_span = (timestamps.max() - timestamps.min()).total_seconds() / 60
            return float(time_span)
            
        except Exception as e:
            logger.error(f"Error calculating time span: {e}")
            return 0.0

def main():
    """
    Example usage of the BiometricFeatureExtractor.
    """
    # Create sample data
    np.random.seed(42)
    timestamps = pd.date_range('2025-01-01', periods=100, freq='1min')
    heart_rate = 70 + 10 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 5, 100)
    blood_pressure = 120 + 5 * np.cos(np.linspace(0, 2*np.pi, 100)) + np.random.normal(0, 3, 100)
    
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure
    })
    
    # Extract features
    extractor = BiometricFeatureExtractor()
    features = extractor.extract_comprehensive_features(sample_data)
    
    print("Extracted Features:")
    for key, value in sorted(features.items()):
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
