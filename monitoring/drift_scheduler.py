"""
Drift Monitoring Scheduler - Runs periodic drift checks
"""
import os
import time
import logging
from datetime import datetime
from drift_detector import MonitoringService
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 3600))
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "/app/logs/predictions.jsonl")

def run_monitoring_loop():
    config = {
        'prediction_log_path': PREDICTION_LOG_PATH,
        'baseline_metrics': {
            'approval_rate': 0.69,
            'avg_confidence': 0.85
        },
        'data_drift_threshold': 0.05
    }
    
    monitor = MonitoringService(config)
    
    logger.info(f"Starting drift monitoring service (interval: {CHECK_INTERVAL}s)")
    
    while True:
        try:
            logger.info(f"Running drift check at {datetime.utcnow().isoformat()}")
            results = monitor.run_monitoring_check()
            
            logger.info(f"Monitoring results: status={results['overall_status']}")
            
            if results['alerts']:
                for alert in results['alerts']:
                    logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")
            
            with open("/app/logs/monitoring_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error during monitoring check: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_monitoring_loop()
