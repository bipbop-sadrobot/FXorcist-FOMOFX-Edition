from typing import Dict, Any
from forex_ai_dashboard.reinforcement import memory_matrix, memory_schema
from forex_ai_dashboard.reinforcement.shared_metadata import SharedMetadata
import graphviz
import json
from datetime import datetime

class DocumentationGenerator:
    """Automatically generates system documentation"""
    
    def __init__(self):
        self.metadata = SharedMetadata()
        
    def generate_architecture_diagram(self) -> graphviz.Digraph:
        """Generate architecture diagram using Graphviz"""
        dot = graphviz.Digraph(comment='Memory System Architecture')
        
        # Core components
        dot.node('A', 'Memory Matrix')
        dot.node('B', 'Model Tracker')
        dot.node('C', 'Event Bus')
        dot.node('D', 'Shared Metadata')
        
        # Relationships
        dot.edges(['AB', 'AC', 'AD', 'BC', 'BD', 'CD'])
        
        # Subsystems
        with dot.subgraph(name='cluster_0') as c:
            c.attr(label='Reinforcement Learning')
            c.node('E', 'Hierarchical RL')
            c.node('F', 'Meta-Controller')
            c.edges(['EF', 'EA', 'EB'])
            
        dot.attr(label=f"\nGenerated at {datetime.now().isoformat()}")
        return dot
        
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system documentation"""
        return {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "memory_matrix": self._get_memory_stats(),
                "model_tracker": self._get_tracker_stats(),
                "event_bus": self._get_bus_stats()
            },
            "metadata": self.metadata.generate_documentation()
        }
        
    def _get_memory_stats(self) -> Dict[str, Any]:
        return {
            "record_count": len(memory_schema.MemorySchema().records),
            "average_error": memory_matrix.MemoryMatrix().calculate_meta_features().get('avg_error')
        }
        
    def _get_tracker_stats(self) -> Dict[str, Any]:
        return {
            "tracked_models": list(set(r.model_version for r in memory_schema.MemorySchema().records))
        }
        
    def _get_bus_stats(self) -> Dict[str, Any]:
        return {
            "event_types": ["new_prediction", "model_registered", "memory_updated"]
        }
        
    def save_documentation(self, path: str):
        """Save complete documentation package"""
        report = self.generate_system_report()
        with open(f"{path}/system_report.json", 'w') as f:
            json.dump(report, f)
            
        self.generate_architecture_diagram().render(f"{path}/architecture")
