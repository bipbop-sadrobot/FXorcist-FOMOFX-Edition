"""
Backward compatibility shim for run_enhanced_training.py
Redirects to the new package structure.
"""

from fxorcist.models.runner import main

if __name__ == '__main__':
    main()