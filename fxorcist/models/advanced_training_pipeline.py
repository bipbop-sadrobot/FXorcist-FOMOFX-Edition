"""
Backward compatibility shim for advanced_training_pipeline.py
Redirects to the new package structure.
"""

from fxorcist.models.advanced_training import AdvancedTrainingPipeline

if __name__ == '__main__':
    pipeline = AdvancedTrainingPipeline()
    pipeline.run_pipeline()