#!/usr/bin/env python3
"""
Simple Verbose Forex Training Runner
====================================
Clear terminal execution with verbose output and progress tracking.

Usage:
    python run_verbose_training.py

Features:
- Verbose progress updates
- Real-time training insights
- User-friendly graphics
- Clear terminal execution
- Performance monitoring
"""

import os
import sys
import time
import argparse
from pathlib import Path

def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print("🎯 VERBOSE FOREX TRAINING SYSTEM")
    print("="*80)
    print("📊 Training Duration: 10 minutes (600 seconds)")
    print("🎨 Verbose Output: Enabled")
    print("📈 Progress Tracking: Real-time")
    print("🖼️  Visualizations: User-friendly graphics")
    print("="*80)

def check_requirements():
    """Check if required files exist"""
    print("\n🔍 CHECKING REQUIREMENTS...")

    required_files = [
        'ultimate_comprehensive_forex_training_all_indicators.py',
        'data/processed'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all required files are present.")
        return False

    print("✅ All requirements satisfied")
    return True

def run_training():
    """Run the verbose training"""
    print("\n🚀 STARTING VERBOSE TRAINING...")

    # Import the training module
    try:
        print("📦 Importing training module...")
        from ultimate_comprehensive_forex_training_all_indicators import VerboseForexTrainer
        print("✅ Module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import training module: {e}")
        return False

    # Create trainer instance
    print("🏗️  Initializing trainer...")
    trainer = VerboseForexTrainer()
    print("✅ Trainer initialized")

    # Run training
    print("\n🎯 EXECUTING TRAINING PIPELINE...")
    print("=" * 60)

    start_time = time.time()

    try:
        trainer.run_verbose_training(
            data_path="data/processed",
            save_visualizations=True
        )

        total_time = time.time() - start_time
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(".2f")
        return True

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 USAGE EXAMPLES:")
    print("-" * 40)
    print("1. Basic execution:")
    print("   python run_verbose_training.py")
    print()
    print("2. With custom data path:")
    print("   python run_verbose_training.py --data-path /path/to/data")
    print()
    print("3. Skip visualizations:")
    print("   python run_verbose_training.py --no-visualizations")
    print()
    print("4. Quiet mode:")
    print("   python run_verbose_training.py --quiet")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verbose Forex Training Runner')
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='Path to data directory')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples')

    args = parser.parse_args()

    if args.examples:
        show_usage_examples()
        return

    # Print header
    print_header()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Run training
    success = run_training()

    if success:
        print("\n🎯 VERBOSE TRAINING COMPLETED!")
        print("📊 Check the logs/ directory for detailed logs")
        print("📈 Check the visualizations/ directory for charts")
        print("🤖 Check the models/trained/ directory for saved models")
    else:
        print("\n❌ VERBOSE TRAINING FAILED!")
        print("📋 Check the error messages above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()