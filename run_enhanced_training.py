#!/usr/bin/env python3
"""
Enhanced Training Runner
Simple script to run the comprehensive training pipeline with various options.
"""

import sys
import os
from pathlib import Path
import argparse

# Add current directory to path
sys.path.append('.')

def main():
    """Run enhanced training with different configurations."""
    parser = argparse.ArgumentParser(description="Enhanced Forex AI Training Runner")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Training mode: 1=basic, 2=optimized, 3=full, 4=custom"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Include ensemble methods"
    )
    parser.add_argument(
        "--interpretability",
        action="store_true",
        help="Enable model interpretability"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=30,
        help="Number of features to select"
    )

    args = parser.parse_args()

    print("🤖 Enhanced Forex AI Training Runner")
    print("=" * 50)

    if args.mode == 1:
        print("\n🚀 Running basic training...")
        cmd = "python comprehensive_training_pipeline.py --no-selection"

    elif args.mode == 2:
        print("\n🚀 Running optimized training...")
        cmd = f"python comprehensive_training_pipeline.py --optimize --features {args.features}"

    elif args.mode == 3:
        print("\n🚀 Running full comprehensive training...")
        cmd = f"python comprehensive_training_pipeline.py --optimize --ensemble --interpretability --features {args.features}"

    elif args.mode == 4:
        print("\n🔧 Custom configuration:")
        cmd = "python comprehensive_training_pipeline.py"
        if args.optimize:
            cmd += " --optimize"
        if args.ensemble:
            cmd += " --ensemble"
        if args.interpretability:
            cmd += " --interpretability"
        cmd += f" --features {args.features}"

    print(f"Command: {cmd}")
    result = os.system(cmd)

    if result == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed with exit code: {result}")

    print("\n" + "=" * 50)
    print("Check the following files:")
    print("• logs/comprehensive_training_*.log - Training logs")
    print("• logs/comprehensive_training_results_*.json - Detailed results")
    print("• logs/comprehensive_training_summary_*.txt - Summary report")
    print("• models/trained/ - Saved models")
    print("=" * 50)

if __name__ == "__main__":
    main()