# LLMEyesim
[![Paper](https://img.shields.io/badge/Paper-View-green?style=flat&logo=adobeacrobatreader)](https://arxiv.org/abs/2408.03515)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

![img](imgs/img.png)

## Installation

- Download [eyesim](https://roblab.org/eyesim/)
- create `config.yml` and set the content according to the `config.example.yml`
- set up conda environment
```bash
conda env create -f environment.yml
conda activate llmeyesim
pip install -e .
```
- Execute the demo
```bash
llmeyesim
```

- Parameters for V1
  - `--world`: `demo`, `free`, `static`, `dynamic`, `mixed`
  - `--model`: `gpt-4o`, `gpt-4o-mini`
  - `--attack`: `none`, `omi`, `ghi`
  - `--defense`: `false`, `true`
  - `--attack rate`: `0.1`, `0.3`, `0.5`, `0.7`, `1`

- Parameters for V2
  - TBD