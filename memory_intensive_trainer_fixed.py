#!/usr/bin/env python3
"""
MEMORY INTENSIVE FOREX TRAINER - Multiple GB RAM Usage
======================================================
Simple, effective memory consumption for testing purposes.

🎯 FEATURES:
├── Massive DataFrame Creation: Creates GBs of forex data
├── Multiple Array Storage: Keeps large arrays in memory
├── Parallel Memory Operations: Multi-threaded memory usage
├── Memory Monitoring: Real-time usage tracking
├── Configurable Memory Targets: Set exact GB targets

🚀 USAGE:
    python memory_intensive_trainer_fixed.py --target-gb 4
    python memory_intensive_trainer_fixed.py --target-gb 8 --parallel 8
    python memory_intensive_trainer_fixed.py --target-gb 2 --no-gc

Author: Kilo Code - Memory Intensive Edition
Version: 1.0.0
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import os
import gc
import time
import threading
import multiprocessing as mp
from tqdm import tqdm
import argparse
import logging

class MemoryIntensiveTrainer:
    """Memory-intensive trainer that consumes multiple GBs of RAM"""

    def __init__(self, target_gb: int = 2, parallel_processes: int = 4, disable_gc: bool = False):
        self.target_gb = target_gb
        self.parallel_processes = parallel_processes
        self.disable_gc = disable_gc

        # Memory storage
        self.dataframes = []
        self.arrays = []
        self.matrices = []

        # Memory tracking
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = 0

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("MemoryTrainer")

        # Disable GC if requested
        if disable_gc:
            gc.disable()
            self.logger.info("Garbage collection disabled")

    def run_memory_intensive_training(self):
        """Run memory-intensive training operations"""
        print("🚀 MEMORY INTENSIVE FOREX TRAINER")
        print("=" * 50)
        print(f"🎯 Target Memory: {self.target_gb}GB")
        print(f"⚡ Parallel Processes: {self.parallel_processes}")
        print(f"🗑️  GC Disabled: {self.disable_gc}")
        print("=" * 50)

        try:
            # Phase 1: Create massive DataFrames
            self._create_massive_dataframes()

            # Phase 2: Generate large NumPy arrays
            self._create_large_arrays()

            # Phase 3: Create memory matrices
            self._create_memory_matrices()

            # Phase 4: Parallel memory operations
            self._run_parallel_operations()

            # Phase 5: Memory report
            self._print_memory_report()

            # Phase 6: Cleanup
            if not self.disable_gc:
                self._cleanup()

        except Exception as e:
            self.logger.error(f"Memory training failed: {e}")
            if not self.disable_gc:
                gc.enable()
            raise

    def _create_massive_dataframes(self):
        """Create massive DataFrames to consume memory"""
        print("\n📊 PHASE 1: Creating Massive DataFrames")

        # Calculate target rows (rough estimate: 50MB per 100k rows)
        target_rows = (self.target_gb * 1024**3) // (50 * 1024**2) * 1000
        target_rows = max(target_rows, 50000)  # Minimum 50k rows

        print(f"Target: {target_rows:,} rows per DataFrame")

        # Create multiple large DataFrames
        for i in range(3):
            print(f"Creating DataFrame {i+1}/3...")

            # Generate datetime index
            start_date = datetime(2020, 1, 1)
            dates = pd.date_range(start_date, periods=target_rows, freq='1min')

            # Create large DataFrame with many columns
            df_data = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_data[col] = np.random.normal(1.0, 0.1, target_rows)

            # Add many technical indicator columns
            for j in range(50):  # 50 additional columns
                df_data[f'indicator_{j}'] = np.random.normal(0, 1, target_rows)

            df = pd.DataFrame(df_data, index=dates)
            self.dataframes.append(df)

            # Force memory usage by keeping reference
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            print(".1f")

    def _create_large_arrays(self):
        """Create large NumPy arrays"""
        print("\n💾 PHASE 2: Creating Large NumPy Arrays")

        # Calculate array size for target memory
        bytes_per_float64 = 8
        total_elements = (self.target_gb * 1024**3) // bytes_per_float64
        array_size = int(np.sqrt(total_elements // 4))  # 4 arrays total

        print(f"Array size: {array_size}x{array_size} ({array_size**2 * 8 / 1024**3:.2f}GB per array)")

        for i in tqdm(range(4), desc="Creating arrays"):
            # Create large array
            large_array = np.random.normal(0, 1, (array_size, array_size)).astype(np.float64)
            self.arrays.append(large_array)

            # Perform operations to ensure memory is used
            result = np.dot(large_array, large_array.T)
            self.arrays.append(result)

            # Update peak memory tracking
            current_memory = psutil.virtual_memory().used
            self.peak_memory = max(self.peak_memory, current_memory)

    def _create_memory_matrices(self):
        """Create various memory-intensive matrices"""
        print("\n🧮 PHASE 3: Creating Memory Matrices")

        # Create correlation matrices
        for i, df in enumerate(self.dataframes):
            print(f"Creating correlation matrix {i+1}...")
            corr_matrix = df.corr()
            self.matrices.append(corr_matrix)

        # Create covariance matrices
        for i, df in enumerate(self.dataframes):
            print(f"Creating covariance matrix {i+1}...")
            cov_matrix = df.cov()
            self.matrices.append(cov_matrix)

        # Create large identity matrices
        size = 5000
        for i in range(3):
            print(f"Creating identity matrix {i+1} ({size}x{size})...")
            identity = np.eye(size, dtype=np.float64)
            self.matrices.append(identity)

    def _run_parallel_operations(self):
        """Run parallel memory-intensive operations"""
        print(f"\n⚡ PHASE 4: Running {self.parallel_processes} Parallel Operations")

        def memory_task(task_id: int):
            """Memory-intensive task"""
            # Create task-specific large arrays
            size = 2000
            arrays = []

            for i in range(5):
                arr = np.random.normal(0, 1, (size, size)).astype(np.float64)
                arrays.append(arr)

                # Perform matrix operations
                if len(arrays) > 1:
                    result = np.dot(arrays[-1], arrays[-2])
                    arrays.append(result)

            # Calculate memory usage
            total_memory = sum(arr.nbytes for arr in arrays)

            time.sleep(0.5)  # Simulate processing time

            return {
                'task_id': task_id,
                'arrays_created': len(arrays),
                'memory_used': total_memory / 1024**2  # MB
            }

        # Run parallel tasks
        with mp.Pool(processes=self.parallel_processes) as pool:
            results = []
            for i in range(self.parallel_processes):
                result = pool.apply_async(memory_task, (i,))
                results.append(result)

            # Collect results
            for result in tqdm(results, desc="Parallel processing"):
                task_result = result.get()
                print(f"  ✅ Task {task_result['task_id']}: {task_result['memory_used']:.1f}MB used")

    def _print_memory_report(self):
        """Print comprehensive memory usage report"""
        print("\n📈 MEMORY USAGE REPORT")
        print("=" * 50)

        # Current memory stats
        current_memory = psutil.virtual_memory()
        process = psutil.Process()

        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")

        # DataFrame memory
        df_memory = sum(df.memory_usage(deep=True).sum() for df in self.dataframes) / 1024**3
        print(".2f")

        # Array memory
        array_memory = sum(arr.nbytes for arr in self.arrays) / 1024**3
        print(".2f")

        # Matrix memory
        matrix_memory = sum(matrix.memory_usage(deep=True).sum() if hasattr(matrix, 'memory_usage')
                           else matrix.nbytes for matrix in self.matrices) / 1024**3
        print(".2f")

        # Total memory used by this process
        total_used = (current_memory.used - self.start_memory) / 1024**3
        print(".2f")

        print(f"📊 DataFrames: {len(self.dataframes)}")
        print(f"📊 Arrays: {len(self.arrays)}")
        print(f"📊 Matrices: {len(self.matrices)}")

        print("=" * 50)

        # Performance summary
        print("🎯 PERFORMANCE SUMMARY")
        print(".2f")
        print(f"⚡ Parallel Processes: {self.parallel_processes}")
        print(f"🗑️  GC Disabled: {self.disable_gc}")

        if total_used >= self.target_gb:
            print("✅ Target memory usage achieved!")
        else:
            print(f"⚠️  Memory usage below target: {total_used:.2f}GB vs {self.target_gb}GB target")

    def _cleanup(self):
        """Perform memory cleanup"""
        print("\n🧹 PHASE 6: Memory Cleanup")

        # Clear all stored objects
        self.dataframes.clear()
        self.arrays.clear()
        self.matrices.clear()

        # Force garbage collection
        collected = gc.collect()
        print(f"🗑️  Garbage collected: {collected} objects")

        # Final memory check
        final_memory = psutil.virtual_memory()
        print(".1f")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Memory Intensive Forex Trainer - Multiple GB RAM Usage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Memory Usage Examples:
  python memory_intensive_trainer_fixed.py --target-gb 2
  python memory_intensive_trainer_fixed.py --target-gb 4 --parallel 8
  python memory_intensive_trainer_fixed.py --target-gb 8 --no-gc
  python memory_intensive_trainer_fixed.py --target-gb 1 --parallel 2

Memory Targets:
  --target-gb N        Target memory usage in GB (default: 2)
  --parallel N         Number of parallel processes (default: 4)
  --no-gc             Disable automatic garbage collection
        """
    )

    parser.add_argument('--target-gb', type=int, default=2,
                        help='Target memory usage in GB (default: 2)')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel processes (default: 4)')
    parser.add_argument('--no-gc', action='store_true',
                        help='Disable automatic garbage collection')

    args = parser.parse_args()

    print(f"🚀 Starting Memory Intensive Trainer")
    print(f"🎯 Target: {args.target_gb}GB")
    print(f"⚡ Parallel: {args.parallel}")
    print(f"🗑️  No GC: {args.no_gc}")
    print("-" * 40)

    # Create and run trainer
    trainer = MemoryIntensiveTrainer(
        target_gb=args.target_gb,
        parallel_processes=args.parallel,
        disable_gc=args.no_gc
    )

    trainer.run_memory_intensive_training()

if __name__ == "__main__":
    main()