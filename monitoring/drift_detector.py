"""
Drift Detection Module for ML Model Monitoring
Implements Data Drift and Model Drift Detection
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    CONCEPT_DRIFT = "concept_drift"

@dataclass
class DriftResult:
    feature: str
    drift_type: DriftType
    score: float
    threshold: float
    is_drifted: bool
    details: Dict

class DataDriftDetector:
    """Detects changes in input feature distributions"""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_statistics(reference_data)
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        stats_dict = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                stats_dict[col] = {
                    'type': 'numerical',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'distribution': data[col].values
                }
            else:
                value_counts = data[col].value_counts(normalize=True)
                stats_dict[col] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'unique_values': list(data[col].unique())
                }
        return stats_dict
    
    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        results = []
        current_stats = self._compute_statistics(current_data)
        
        for feature in self.reference_stats:
            if feature not in current_stats:
                continue
            
            ref_stat = self.reference_stats[feature]
            curr_stat = current_stats[feature]
            
            if ref_stat['type'] == 'numerical':
                result = self._detect_numerical_drift(feature, ref_stat, curr_stat)
            else:
                result = self._detect_categorical_drift(feature, ref_stat, curr_stat)
            
            results.append(result)
        
        return results
    
    def _detect_numerical_drift(self, feature: str, ref_stat: Dict, curr_stat: Dict) -> DriftResult:
        ks_stat, p_value = stats.ks_2samp(
            ref_stat['distribution'], 
            curr_stat['distribution']
        )
        
        is_drifted = p_value < self.threshold
        
        return DriftResult(
            feature=feature,
            drift_type=DriftType.DATA_DRIFT,
            score=ks_stat,
            threshold=self.threshold,
            is_drifted=is_drifted,
            details={
                'test': 'Kolmogorov-Smirnov',
                'p_value': p_value,
                'ref_mean': ref_stat['mean'],
                'curr_mean': curr_stat['mean'],
                'ref_std': ref_stat['std'],
                'curr_std': curr_stat['std']
            }
        )
    
    def _detect_categorical_drift(self, feature: str, ref_stat: Dict, curr_stat: Dict) -> DriftResult:
        all_categories = set(ref_stat['distribution'].keys()) | set(curr_stat['distribution'].keys())
        
        ref_probs = [ref_stat['distribution'].get(cat, 0.0001) for cat in all_categories]
        curr_probs = [curr_stat['distribution'].get(cat, 0.0001) for cat in all_categories]
        
        ref_probs = np.array(ref_probs) / sum(ref_probs)
        curr_probs = np.array(curr_probs) / sum(curr_probs)
        
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        is_drifted = psi > 0.2
        
        return DriftResult(
            feature=feature,
            drift_type=DriftType.DATA_DRIFT,
            score=psi,
            threshold=0.2,
            is_drifted=is_drifted,
            details={
                'test': 'Population Stability Index (PSI)',
                'psi_value': psi,
                'interpretation': 'No drift' if psi < 0.1 else 'Moderate drift' if psi < 0.2 else 'Significant drift'
            }
        )

class ModelDriftDetector:
    """Detects changes in model performance and predictions"""
    
    def __init__(self, baseline_metrics: Dict, prediction_log_path: str):
        self.baseline_metrics = baseline_metrics
        self.prediction_log_path = Path(prediction_log_path)
        self.performance_threshold = 0.05
    
    def detect_prediction_drift(self, window_hours: int = 24) -> DriftResult:
        predictions = self._load_recent_predictions(window_hours)
        
        if len(predictions) < 100:
            return DriftResult(
                feature="prediction_distribution",
                drift_type=DriftType.MODEL_DRIFT,
                score=0.0,
                threshold=0.1,
                is_drifted=False,
                details={'message': 'Insufficient data for drift detection'}
            )
        
        approval_rate = sum(1 for p in predictions if p['prediction'] == 'Y') / len(predictions)
        baseline_rate = self.baseline_metrics.get('approval_rate', 0.69)
        
        drift_score = abs(approval_rate - baseline_rate)
        is_drifted = drift_score > 0.1
        
        return DriftResult(
            feature="prediction_distribution",
            drift_type=DriftType.MODEL_DRIFT,
            score=drift_score,
            threshold=0.1,
            is_drifted=is_drifted,
            details={
                'current_approval_rate': approval_rate,
                'baseline_approval_rate': baseline_rate,
                'sample_size': len(predictions),
                'window_hours': window_hours
            }
        )
    
    def detect_confidence_drift(self, window_hours: int = 24) -> DriftResult:
        predictions = self._load_recent_predictions(window_hours)
        
        if len(predictions) < 100:
            return DriftResult(
                feature="prediction_confidence",
                drift_type=DriftType.MODEL_DRIFT,
                score=0.0,
                threshold=0.15,
                is_drifted=False,
                details={'message': 'Insufficient data'}
            )
        
        probabilities = [p['probability'] for p in predictions]
        avg_confidence = np.mean([max(p, 1-p) for p in probabilities])
        baseline_confidence = self.baseline_metrics.get('avg_confidence', 0.85)
        
        drift_score = abs(avg_confidence - baseline_confidence)
        is_drifted = drift_score > 0.15
        
        return DriftResult(
            feature="prediction_confidence",
            drift_type=DriftType.MODEL_DRIFT,
            score=drift_score,
            threshold=0.15,
            is_drifted=is_drifted,
            details={
                'current_avg_confidence': avg_confidence,
                'baseline_avg_confidence': baseline_confidence,
                'sample_size': len(predictions)
            }
        )
    
    def _load_recent_predictions(self, window_hours: int) -> List[Dict]:
        predictions = []
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        
        try:
            if self.prediction_log_path.exists():
                with open(self.prediction_log_path, 'r') as f:
                    for line in f:
                        try:
                            pred = json.loads(line.strip())
                            pred_time = datetime.fromisoformat(pred['timestamp'])
                            if pred_time >= cutoff_time:
                                predictions.append(pred)
                        except:
                            continue
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
        
        return predictions

class MonitoringService:
    """Main monitoring service that orchestrates drift detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alerts = []
        
    def run_monitoring_check(self, current_data: Optional[pd.DataFrame] = None) -> Dict:
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_drift': [],
            'model_drift': [],
            'alerts': [],
            'overall_status': 'healthy'
        }
        
        if current_data is not None and 'reference_data' in self.config:
            data_detector = DataDriftDetector(
                self.config['reference_data'],
                threshold=self.config.get('data_drift_threshold', 0.05)
            )
            data_drift_results = data_detector.detect_drift(current_data)
            results['data_drift'] = [self._result_to_dict(r) for r in data_drift_results]
            
            drifted_features = [r.feature for r in data_drift_results if r.is_drifted]
            if drifted_features:
                results['alerts'].append({
                    'type': 'DATA_DRIFT',
                    'severity': 'warning',
                    'message': f"Data drift detected in features: {', '.join(drifted_features)}"
                })
        
        if 'prediction_log_path' in self.config:
            model_detector = ModelDriftDetector(
                baseline_metrics=self.config.get('baseline_metrics', {}),
                prediction_log_path=self.config['prediction_log_path']
            )
            
            pred_drift = model_detector.detect_prediction_drift()
            conf_drift = model_detector.detect_confidence_drift()
            
            results['model_drift'] = [
                self._result_to_dict(pred_drift),
                self._result_to_dict(conf_drift)
            ]
            
            if pred_drift.is_drifted:
                results['alerts'].append({
                    'type': 'MODEL_DRIFT',
                    'severity': 'critical',
                    'message': f"Prediction distribution drift detected: {pred_drift.details}"
                })
            
            if conf_drift.is_drifted:
                results['alerts'].append({
                    'type': 'CONFIDENCE_DRIFT',
                    'severity': 'warning',
                    'message': f"Model confidence drift detected: {conf_drift.details}"
                })
        
        if any(a['severity'] == 'critical' for a in results['alerts']):
            results['overall_status'] = 'critical'
        elif results['alerts']:
            results['overall_status'] = 'warning'
        
        return results
    
    def _result_to_dict(self, result: DriftResult) -> Dict:
        return {
            'feature': result.feature,
            'drift_type': result.drift_type.value,
            'score': float(result.score),
            'threshold': result.threshold,
            'is_drifted': result.is_drifted,
            'details': result.details
        }

if __name__ == "__main__":
    np.random.seed(42)
    reference_df = pd.DataFrame({
        'ApplicantIncome': np.random.normal(5000, 2000, 1000),
        'LoanAmount': np.random.normal(150, 50, 1000),
        'Credit_History': np.random.choice([0, 1], 1000, p=[0.15, 0.85])
    })
    
    current_df = pd.DataFrame({
        'ApplicantIncome': np.random.normal(5500, 2500, 500),
        'LoanAmount': np.random.normal(160, 60, 500),
        'Credit_History': np.random.choice([0, 1], 500, p=[0.25, 0.75])
    })
    
    detector = DataDriftDetector(reference_df)
    results = detector.detect_drift(current_df)
    
    for result in results:
        print(f"Feature: {result.feature}")
        print(f"  Drifted: {result.is_drifted}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Details: {result.details}")
        print()
