# Agentic Ablation

A framework for automated code ablation studies using LLM agents. This project helps analyze the importance of different components in neural network architectures through systematic removal and testing.

![Ablation](/sources/ablation_study.png)
## Overview

Agentic Ablation uses a multi-agent workflow to automatically:
1. Analyze code with neural network architectures
2. Generate ablated versions (with specific components removed)
3. Test the modified code to ensure it remains functional
4. Analyze the impact of removals on model performance

## Key Features

- **Automated Ablation**: Identifies components marked with `#ABLATABLE_COMPONENT` comments
- **Multi-Agent System**: Specialized agents for code generation, execution, reflection, and analysis
- **Failure Recovery**: Built-in reflection and retry mechanisms for robust execution
- **Visualization**: Generates comparison plots between original and ablated models
- **Result Analysis**: Provides detailed insights on the impact of ablated components

## Getting Started

### Prerequisites

- Python 3.13+
- OpenAI API key (for LLM agents)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-ablation.git
cd agentic-ablation

# Install dependencies with uv (using pyproject.toml)
uv pip install -e .
```

### Usage

1. Mark ablatable components in your neural network code:
```python
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #ABLATABLE_COMPONENT
```

2. Run the ablation study:
```bash
make run-agent
```
   This will use the `uv run` command defined in the Makefile.

3. View results in the generated JSON files and PDF reports.

## Project Structure

The framework is organized into specialized modules:
- `agents/`: Implementation of each specialized agent
- `models/`: Data schemas for code and analysis
- `workflow/`: LangGraph-based workflow configuration
- `utils/`: Helper functions for file operations
- `prompts/`: LLM prompts for each agent

## License

MIT

## Acknowledgments

Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph).