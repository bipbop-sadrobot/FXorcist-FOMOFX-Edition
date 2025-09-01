from typing import Dict, List, Any
from forex_ai_dashboard.reinforcement.integrated_memory import IntegratedMemorySystem
from forex_ai_dashboard.reinforcement.event_bus import EventBus
from forex_ai_dashboard.reinforcement.shared_metadata import SharedMetadata
from forex_ai_dashboard.utils.logger import logger
import hashlib

class DistributedMemorySystem:
    """Manages sharded memory instances across cluster"""
    
    def __init__(self, event_bus: EventBus, metadata: SharedMetadata, shard_count: int = 3):
        self.event_bus = event_bus
        self.metadata = metadata
        self.shard_count = shard_count
        self.shards = {
            f"shard_{i}": IntegratedMemorySystem(event_bus, metadata)
            for i in range(shard_count)
        }
        
    def get_shard(self, key: str) -> IntegratedMemorySystem:
        """Get appropriate shard based on hashed key"""
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_val % self.shard_count
        return self.shards[f"shard_{shard_index}"]
        
    def add_record(self, record: Dict[str, Any]):
        """Add record to appropriate shard"""
        shard = self.get_shard(record['model_version'])
        shard.add_record(record)
        
    def analyze_cluster_trends(self) -> Dict[str, Any]:
        """Aggregate analysis across all shards"""
        combined_analysis = {}
        for shard_name, shard in self.shards.items():
            analysis = shard.analyze_memory_trends()
            for key, value in analysis.items():
                if isinstance(value, (int, float)):
                    combined_analysis.setdefault(key, 0)
                    combined_analysis[key] += value
                elif isinstance(value, dict):
                    combined_analysis.setdefault(key, {})
                    combined_analysis[key].update({f"{shard_name}_{k}": v for k,v in value.items()})
                    
        # Calculate averages
        if 'total_records' in combined_analysis:
            combined_analysis['avg_records_per_shard'] = combined_analysis['total_records'] / self.shard_count
            
        return combined_analysis
        
    def rebalance_shards(self):
        """Rebalance records across shards based on current load"""
        logger.info("Initiating shard rebalancing")
        all_records = []
        for shard in self.shards.values():
            all_records.extend(shard.rolling_memory.records)
            shard.rolling_memory.records = []
            
        for record in all_records:
            self.add_record(record)
