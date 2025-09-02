#!/usr/bin/env python3
"""
Enhanced Training Runner
Simple script to run the comprehensive training pipeline with various options.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def main():
    """Run enhanced training with different configurations."""

    print("🤖 Enhanced Forex AI Training Runner")
    print("=" * 50)

    print("\nAvailable training options:")
    print("1. Basic training (fast, no optimization)")
    print("2. Optimized training (with hyperparameter tuning)")
    print("3. Full comprehensive training (all features)")
    print("4. Custom configuration")

    while True:
        try:
            choice = input("\nSelect training mode (1-4): ").strip()

            if choice == "1":
                print("\n🚀 Running basic training...")
                os.system("python comprehensive_training_pipeline.py --no-selection")

            elif choice == "2":
                print("\n🚀 Running optimized training...")
                os.system("python comprehensive_training_pipeline.py --optimize --features 30")

            elif choice == "3":
                print("\n🚀 Running full comprehensive training...")
                os.system("python comprehensive_training_pipeline.py --optimize --ensemble --interpretability --features 50")

            elif choice == "4":
                print("\n🔧 Custom configuration:")
                optimize = input("Enable hyperparameter optimization? (y/n): ").lower() == 'y'
                ensemble = input("Include ensemble methods? (y/n): ").lower() == 'y'
                interpret = input("Enable interpretability? (y/n): ").lower() == 'y'
                features = int(input("Number of features to select: "))

                cmd = "python comprehensive_training_pipeline.py"
                if optimize:
                    cmd += " --optimize"
                if ensemble:
                    cmd += " --ensemble"
                if interpret:
                    cmd += " --interpretability"
                cmd += f" --features {features}"

                print(f"\n🚀 Running custom training: {cmd}")
                os.system(cmd)

            else:
                print("Invalid choice. Please select 1-4.")
                continue

            break

        except KeyboardInterrupt:
            print("\n👋 Training cancelled by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    print("\n" + "=" * 50)
    print("Training completed! Check the following files:")
    print("• logs/comprehensive_training_*.log - Training logs")
    print("• logs/comprehensive_training_results_*.json - Detailed results")
    print("• logs/comprehensive_training_summary_*.txt - Summary report")
    print("• models/trained/ - Saved models")
    print("=" * 50)

if __name__ == "__main__":
    main()