"""
Simple Experiment Tracking Utility
Replaces MLflow for lightweight experiment tracking
"""

import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """Simple experiment tracker to replace MLflow"""
    
    def __init__(self, experiment_name="default_experiment"):
        self.experiment_name = experiment_name
        self.runs_dir = Path("experiments") / experiment_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.current_run = None
        self.run_data = {}
    
    def start_run(self, run_name=None):
        """Start a new experiment run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = run_name
        self.run_data = {
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
            'parameters': {},
            'metrics': {},
            'model_info': {}
        }
        print(f"Started run: {run_name}")
        return self
    
    def log_params(self, params):
        """Log hyperparameters"""
        if isinstance(params, dict):
            self.run_data['parameters'].update(params)
        return self
    
    def log_metrics(self, metrics):
        """Log performance metrics"""
        if isinstance(metrics, dict):
            # Convert numpy types to native Python types for JSON serialization
            self.run_data['metrics'].update({
                k: float(v) if hasattr(v, '__float__') else v 
                for k, v in metrics.items()
            })
        return self
    
    def log_model_info(self, model_type, model_path=None):
        """Log model information"""
        self.run_data['model_info'] = {
            'model_type': model_type,
            'model_path': model_path,
            'saved_at': datetime.now().isoformat()
        }
        return self
    
    def end_run(self):
        """End current run and save to file"""
        if self.current_run is None:
            print("No active run to end")
            return
        
        self.run_data['end_time'] = datetime.now().isoformat()
        
        # Save run data to JSON
        run_file = self.runs_dir / f"{self.current_run}.json"
        with open(run_file, 'w') as f:
            json.dump(self.run_data, f, indent=2)
        
        print(f"Ended run: {self.current_run}")
        print(f"Results saved to: {run_file}")
        
        self.current_run = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_run is not None:
            self.end_run()
        return False


# Global experiment tracker instance
_tracker = None


def set_experiment(experiment_name):
    """Set the current experiment (similar to mlflow.set_experiment)"""
    global _tracker
    _tracker = ExperimentTracker(experiment_name)
    print(f"Experiment '{experiment_name}' initialized!")
    return _tracker


def start_run(run_name=None):
    """Start a run (similar to mlflow.start_run)"""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker.start_run(run_name)


def log_params(params):
    """Log parameters"""
    global _tracker
    if _tracker is not None and _tracker.current_run is not None:
        _tracker.log_params(params)
    else:
        print("Warning: No active run. Create a run first.")


def log_metrics(metrics):
    """Log metrics"""
    global _tracker
    if _tracker is not None and _tracker.current_run is not None:
        _tracker.log_metrics(metrics)
    else:
        print("Warning: No active run. Create a run first.")


# Context manager support
class RunContext:
    """Context manager for experiment runs"""
    
    def __init__(self, run_name=None):
        self.run_name = run_name
        self.tracker = None
    
    def __enter__(self):
        global _tracker
        if _tracker is None:
            _tracker = ExperimentTracker()
        self.tracker = _tracker.start_run(self.run_name)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker is not None:
            self.tracker.end_run()
        return False


def start_run_context(run_name=None):
    """Create a context manager for a run (similar to mlflow.start_run as context manager)"""
    return RunContext(run_name)

