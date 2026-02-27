# Cerebral SDK Prototype

Neuromorphic LLM orchestration prototype with brain-inspired modules: PFC, Hippocampus, Thalamus, Wernicke, Corpus Callosum with ADHD-optimized event scoring.

## Architecture

### 🧠 Brain-Inspired Modules

#### Prefrontal Cortex (PFC)
- **Purpose**: Short-term operational memory
- **Features**: 
  - `NeuralEvent` class with embeddings, significance, novelty, emotional valence
  - Salience computation and consolidation thresholds
  - Capacity management with automatic pruning
  - Insula-like decay for significance-based memory pruning

#### Hippocampus
- **Purpose**: Semantic long-term memory with embeddings
- **Features**:
  - Sentence-transformers for vector embeddings (MiniLM)
  - Cosine similarity search for memory retrieval
  - Pattern completion via similarity thresholds
  - ADHD cognitive load management via pruning

#### Wernicke's Area
- **Purpose**: Context compression and summarization
- **Features**:
  - Token-efficient context fitting
  - Compression ratio control
  - ADHD-friendly key point extraction
  - OpenAI API integration

#### Thalamus
- **Purpose**: Event scoring and sensory gating
- **Features**:
  - Chaos/Foundation/Glow event classification
  - Keyword-based novelty scoring
  - ADHD-optimized: spike on Glow events, suppress Chaos
  - Context-aware scoring adjustments

#### Corpus Callosum
- **Purpose**: Multi-LLM provider routing
- **Features**:
  - OpenAI and Anthropic provider support
  - Task-based routing rules
  - A/B testing across providers
  - Hemispheric coordination metaphor

## Installation

```bash
# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

## Quick Start

```python
from cerebral_sdk.pfc.neural_event import NeuralEvent, PFCMemory
from cerebral_sdk.thalamus.event_scorer import ThalamusScorer
from cerebral_sdk.hippocampus.semantic_memory import SemanticMemory

# Initialize components
scorer = ThalamusScorer()
pfc = PFCMemory(capacity=50)
hippocampus = SemanticMemory()

# Score and process event
content = "Breakthrough discovery in neuromorphic computing!"
result = scorer.score_event(content, context={'user_initiated': True})

if result['should_process']:
    event = NeuralEvent(
        event_id="event_001",
        content=content,
        novelty=result['novelty'],
        significance=0.8
    )
    pfc.add_event(event)
    
    # Consolidate to long-term memory if significant
    if event.should_consolidate():
        hippocampus.consolidate(event)
```

## Continuous Development Agents

```python
from cerebral_sdk.agents import DevelopmentTask, RepositoryDevelopmentAgent, TaskResult

tasks = [DevelopmentTask(task_id="1", title="Improve test coverage")]

def implement(task: DevelopmentTask) -> TaskResult:
    return TaskResult(task_id=task.task_id, status="implemented", summary=task.title)

agent = RepositoryDevelopmentAgent(implementer=implement)
results = agent.run_continuously(task_source=lambda: tasks, max_cycles=1)
```

## ADHD-Optimized Features

- **Glow Event Spiking**: Amplify breakthrough moments (novelty > 0.8)
- **Chaos Suppression**: Filter routine updates (novelty < 0.3)
- **Significance Decay**: Prune low-priority memories automatically
- **Key Point Extraction**: ADHD-friendly bullet summaries
- **Sparse Activation**: Process only salient events

## Testing

```bash
# Run tests
poetry run pytest cerebral_sdk/tests/

# With coverage
poetry run pytest --cov=cerebral_sdk cerebral_sdk/tests/
```

## Project Structure

```
cerebral-sdk-prototype/
├── cerebral_sdk/
│   ├── pfc/              # Prefrontal Cortex
│   │   └── neural_event.py
│   ├── hippocampus/      # Semantic memory
│   │   └── semantic_memory.py
│   ├── wernicke/         # Summarization
│   │   └── summarizer.py
│   ├── thalamus/         # Event scoring
│   │   └── event_scorer.py
│   ├── corpus_callosum/  # LLM routing
│   │   └── router.py
│   └── tests/
│       └── test_integration.py
├── pyproject.toml
└── README.md
```

## Dependencies

- **Python**: ^3.10
- **openai**: ^1.12.0
- **sentence-transformers**: ^2.3.1
- **torch**: ^2.1.0
- **pydantic**: ^2.5.0

## Philosophy Integration

- **Enactivism**: Predictive coding via Hippocampal pattern completion
- **Cognitivism**: Symbolic event scoring in Thalamus
- **Existentialism**: Meta-state shifts (exploration/exploitation)

## License

MIT
