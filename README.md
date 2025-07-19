# LLMEyesim

[![Paper](https://img.shields.io/badge/Paper-View-green?style=flat&logo=adobeacrobatreader)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5202517)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat&logo=github)](https://github.com/MoeBuTa/LLMEyesim)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

![img](imgs/img.png)

## Overview

LLMEyesim is a research project that integrates Large Language Models (LLMs) with the EyeSim robotics simulator. This project explores the capabilities and vulnerabilities of LLM-powered robotic systems in simulated environments.

## Installation

### Prerequisites
- Download and install [EyeSim](https://roblab.org/eyesim/)
- Python 3.x with conda

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/MoeBuTa/LLMEyesim.git
   cd LLMEyesim
   ```

2. **Configure the environment**
   - Create `config.yml` based on the provided template:
   ```bash
   cp config.example.yml config.yml
   ```
   - Edit `config.yml` with your specific configuration settings

3. **Set up conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate llmeyesim
   pip install -e .
   ```

4. **Run the demo**
   ```bash
   llmeyesim
   ```

## Configuration Parameters

### Version 1 (V1) Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--world` | `demo`, `free`, `static`, `dynamic`, `mixed` | Simulation world type |
| `--model` | `gpt-4o`, `gpt-4o-mini` | LLM model selection |
| `--attack` | `none`, `omi`, `ghi` | Attack type for adversarial testing |
| `--defense` | `false`, `true` | Enable/disable defense mechanisms |
| `--attack-rate` | `0.1`, `0.3`, `0.5`, `0.7`, `1` | Attack frequency rate |

### Version 2 (V2) Parameters
*To be determined - coming soon*

## Usage Examples

```bash
# Basic demo
llmeyesim --world demo --model gpt-4o

# Test with adversarial attacks
llmeyesim --world dynamic --model gpt-4o --attack omi --attack-rate 0.3

# Enable defense mechanisms
llmeyesim --world mixed --model gpt-4o-mini --defense true
```

## Research Paper

For detailed information about the methodology and findings, please refer to our research paper:
[View Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5202517)

## Contributing

This project is currently in active development. Contributions, issues, and feature requests are welcome!

## Contact

For questions or collaboration opportunities, please open an issue on the [GitHub repository](https://github.com/MoeBuTa/LLMEyesim).
