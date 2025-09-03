#!/usr/bin/env python3
"""
SIMPLE MEMORY DEMO - High Memory Usage Demonstration
====================================================
Creates and maintains large data structures in memory to demonstrate GB usage.

🎯 FEATURES:
├── Large DataFrame Creation: Creates massive forex datasets
├── NumPy Array Generation: Produces large computational arrays
├── Memory Monitoring: Real-time memory usage tracking
├── Configurable Targets: Set exact memory consumption goals

🚀 USAGE:
    python simple_memory_demo.py --target-gb 4
    python simple_memory_demo.py --target-gb 8 --no-gc

Author: Kilo Code - Simple Memory Demo
Version: 1.0.0
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from datetime import datetime
import psutil
import gc
import time
import argparse

class SimpleMemoryDemo:
    """Simple demonstration of high memory usage"""

    def __init__(self, target_gb: int = 2, disable_gc: bool = False):
        self.target_gb = target_gb
        self.disable_gc = disable_gc

        # Memory storage containers
        self.dataframes = []
        self.arrays = []

        # Memory tracking
        self.start_memory = psutil.virtual_memory().used

        if disable_gc:
            gc.disable()
            print("🗑️  Garbage collection disabled")

    def run_demo(self):
        """Run the memory demonstration"""
        print("🚀 SIMPLE MEMORY DEMO")
        print("=" * 40)
        print(f"🎯 Target Memory: {self.target_gb}GB")
        print(f"🗑️  GC Disabled: {self.disable_gc}")
        print("=" * 40)

        # Phase 1: Create massive DataFrames
        self._create_dataframes()

        # Phase 2: Create large arrays
        self._create_arrays()

        # Phase 3: Memory report
        self._memory_report()

        # Phase 4: Keep memory allocated (don't exit immediately)
        self._hold_memory()

    def _create_dataframes(self):
        """Create large DataFrames"""
        print("\n📊 Creating Massive DataFrames...")

        # Calculate rows needed for target memory
        rows_per_gb = 500000  # Approximate
        total_rows = self.target_gb * rows_per_gb

        print(f"Target: {total_rows:,} rows")

        # Create multiple large DataFrames
        for i in range(3):
            print(f"  Creating DataFrame {i+1}/3...")

            # Generate data
            dates = pd.date_range('2020-01-01', periods=total_rows//3, freq='1min')

            data = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = np.random.normal(1.0, 0.1, total_rows//3)

            # Add many indicator columns
            for j in range(100):  # 100 columns
                data[f'indicator_{j}'] = np.random.normal(0, 1, total_rows//3)

            df = pd.DataFrame(data, index=dates)
            self.dataframes.append(df)

            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            print(".1f")

    def _create_arrays(self):
        """Create large NumPy arrays"""
        print("\n💾 Creating Large NumPy Arrays...")

        # Calculate array size
        array_elements = (self.target_gb * 1024**3) // 8  # float64 = 8 bytes
        array_size = int(np.sqrt(array_elements // 2))  # 2 arrays

        print(f"Array size: {array_size}x{array_size}")

        for i in range(2):
            print(f"  Creating array {i+1}/2...")
            large_array = np.random.normal(0, 1, (array_size, array_size)).astype(np.float64)
            self.arrays.append(large_array)

            memory_gb = large_array.nbytes / 1024**3
            print(".2f")

    def _memory_report(self):
        """Print memory usage report"""
        print("\n📈 MEMORY USAGE REPORT")
        print("=" * 40)

        current_memory = psutil.virtual_memory()

        # Calculate memory used by this process
        df_memory = sum(df.memory_usage(deep=True).sum() for df in self.dataframes) / 1024**3
        array_memory = sum(arr.nbytes for arr in self.arrays) / 1024**3
        total_used = (current_memory.used - self.start_memory) / 1024**3

        print(".1f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".1f")

        print(f"📊 DataFrames: {len(self.dataframes)}")
        print(f"📊 Arrays: {len(self.arrays)}")

        print("=" * 40)

        if total_used >= self.target_gb:
            print("✅ Target memory usage achieved!")
        else:
            print(f"⚠️  Memory usage: {total_used:.2f}GB (target: {self.target_gb}GB)")

    def _hold_memory(self):
        """Hold memory allocation for observation"""
        print("\n⏳ Holding memory allocation for 30 seconds...")
        print("Monitor system memory usage during this time.")

        for i in range(30, 0, -5):
            current_memory = psutil.virtual_memory()
            used_gb = current_memory.used / 1024**3
            print(".1f")
            time.sleep(5)

        print("\n✅ Memory demonstration complete!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Simple Memory Demo - High Memory Usage Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_memory_demo.py --target-gb 2
  python simple_memory_demo.py --target-gb 4 --no-gc
  python simple_memory_demo.py --target-gb 8

Memory Targets:
  --target-gb N    Target memory usage in GB (default: 2)
  --no-gc         Disable automatic garbage collection
        """
    )

    parser.add_argument('--target-gb', type=int, default=2,
                        help='Target memory usage in GB (default: 2)')
    parser.add_argument('--no-gc', action='store_true',
                        help='Disable automatic garbage collection')

    args = parser.parse_args()

    print(f"🚀 Starting Simple Memory Demo")
    print(f"🎯 Target: {args.target_gb}GB")
    print(f"🗑️  No GC: {args.no_gc}")
    print("-" * 30)

    demo = SimpleMemoryDemo(args.target_gb, args.no_gc)
    demo.run_demo()

if __name__ == "__main__":
    main()