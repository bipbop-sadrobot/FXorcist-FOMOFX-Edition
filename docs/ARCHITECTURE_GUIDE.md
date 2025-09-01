# FXorcist Architecture Guide

## System Overview

```mermaid
graph TD
    Data[Data Sources] --> Ingestion[Data Ingestion]
    Ingestion --> Validation[Data Validation]
    Validation --> Features[Feature Engineering]
    Features --> Training[Model Training]
    Training --> Evaluation[Model Evaluation]
    
    Memory[Memory System] --> Features
    Memory --> Training
    Memory --> Evaluation
    
    Training --> Memory
    Evaluation --> Memory
    
    Dashboard[Dashboard] --> Memory
    Dashboard --> Evaluation
```

## Memory System Architecture

### Memory Layers

```mermaid
graph TD
    API[REST API] --> Manager[Memory Manager]
    Manager --> Vector[Vector Store]
    Manager --> Meta[Metadata Store]
    Manager --> Monitor[Monitoring]
    
    Vector --> FAISS[FAISS Index]
    Meta --> SQLite[SQLite Store]
    Monitor --> Stats[Usage Stats]
    Monitor --> Trends[Trend Analysis]
```

### Memory Tiers

1. Working Memory (WM)
   - Recent predictions and patterns
   - Fast access, limited size
   - Automatic consolidation

2. Long-Term Memory (LTM)
   - Historical patterns
   - Compressed representations
   - Indexed for efficient search

3. Episodic Memory (EM)
   - Significant market events
   - Anomaly records
   - Training feedback

### Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Manager
    participant Storage
    participant Monitor

    Client->>API: Store Entry
    API->>Manager: Process Entry
    Manager->>Storage: Save Data
    Manager->>Monitor: Update Stats
    Monitor->>API: Return Status
```

## Model Hierarchy

### Base Models

```mermaid
graph TD
    Base[BaseModel] --> LSTM[LSTM Model]
    Base --> GNN[GNN Model]
    Base --> TCN[TCN Model]
    Base --> TFT[TFT Model]
```

### Model Integration

```mermaid
graph TD
    Data[Training Data] --> Features[Feature Engineering]
    Features --> Memory[Memory Features]
    Features --> Technical[Technical Indicators]
    
    Memory --> Training[Model Training]
    Technical --> Training
    
    Training --> Evaluation[Model Evaluation]
    Evaluation --> Memory
```

## Pipeline Components

### Data Ingestion

```mermaid
graph TD
    Raw[Raw Data] --> Validation[Data Validation]
    Validation --> Clean[Data Cleaning]
    Clean --> Features[Feature Engineering]
    
    Validation --> Quarantine[Data Quarantine]
    Clean --> Memory[Memory System]
```

### Feature Engineering

1. Technical Features
   - Price indicators
   - Volume metrics
   - Momentum signals

2. Memory Features
   - Historical patterns
   - Market regimes
   - Anomaly indicators

3. Combined Features
   - Feature fusion
   - Cross-validation
   - Importance ranking

### Training Pipeline

```mermaid
graph TD
    Data[Training Data] --> Memory[Memory Check]
    Memory --> Clean[Data Cleaning]
    Clean --> Train[Model Training]
    Train --> Evaluate[Evaluation]
    Evaluate --> Update[Memory Update]
```

## System Components

### Dashboard Architecture

```mermaid
graph TD
    Data[Data Sources] --> Components[Dashboard Components]
    Components --> Status[System Status]
    Components --> Performance[Performance Metrics]
    Components --> Predictions[Model Predictions]
    
    Status --> Memory[Memory Monitor]
    Status --> Resources[Resource Monitor]
    Status --> Pipeline[Pipeline Monitor]
```

### Monitoring System

1. Resource Monitoring
   - CPU/Memory usage
   - Disk utilization
   - Network metrics

2. Memory Monitoring
   - Usage trends
   - Recall latency
   - Tier statistics

3. Pipeline Monitoring
   - Data freshness
   - Processing latency
   - Error rates

## Integration Points

### External Systems

```mermaid
graph TD
    Data[Data Sources] --> Pipeline[Pipeline]
    Pipeline --> Memory[Memory System]
    Memory --> Models[Model Training]
    Models --> Dashboard[Dashboard]
```

### Internal Communication

1. Event System
   - Memory updates
   - Model training
   - System alerts

2. Data Flow
   - Raw data ingestion
   - Feature processing
   - Model predictions

3. Monitoring
   - Health checks
   - Performance metrics
   - Alert system

## Deployment Architecture

### Components

```mermaid
graph TD
    Data[Data Services] --> API[REST API]
    API --> Memory[Memory System]
    Memory --> Models[Model Services]
    API --> Dashboard[Dashboard]
```

### Scaling Considerations

1. Memory System
   - Distributed storage
   - Load balancing
   - Cache layers

2. Model Training
   - Parallel processing
   - Resource allocation
   - Batch optimization

3. Dashboard
   - Component caching
   - Data streaming
   - Resource management

## Security Architecture

1. Data Protection
   - Input validation
   - Data encryption
   - Access control

2. Memory Security
   - Secure storage
   - Access logging
   - Data retention

3. API Security
   - Authentication
   - Rate limiting
   - Request validation

## Future Extensions

1. Distributed Memory
   - Federated storage
   - Sync protocols
   - Conflict resolution

2. Advanced Analytics
   - Pattern mining
   - Trend analysis
   - Anomaly detection

3. Enhanced Integration
   - External APIs
   - Data providers
   - Trading systems