# Teoria de Tudo — Artigo

[![CI](https://github.com/fisicapaulo/teoria_de_tudo_artigo/actions/workflows/ci.yml/badge.svg)](https://github.com/fisicapaulo/teoria_de_tudo_artigo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](#pré-requisitos)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](#testes)
[![Reprodutibilidade](https://img.shields.io/badge/reprodutibilidade-forte-success)](#reprodutibilidade)

Repositório de reprodução e auditoria do artigo:
- Fechamento de \[α^{-1}\] por ICC0 + ICC1
- Certificação espectral de \[m_H\] via fator geométrico-universal

Este repositório segue um protocolo de reprodutibilidade forte: manifests imutáveis, seeds fixadas, precisão decimal explícita e relatórios JSON canônicos com checksums SHA-256.

## Pré-requisitos
- Python 3.11+ (recomendado)
- Git
- Sistemas: Windows, Linux, macOS
- Dependências: ver `requirements.txt`

## Instalação
1) Criar venv:
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.\\.venv\\Scripts\\Activate.ps1`
   - Linux/macOS:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
2) Instalar dependências:
   - `pip install -r requirements.txt`

## Estrutura
- `scripts/`: scripts executáveis (Python) para cálculos, geração de relatórios e auditoria
  - `scripts/constants/c_from_icc.py`: cálculo de \[c\] via \[μ_0\] e \[ε_0\] (ICC-derivado)
  - `scripts/constants/hbar_pipeline.py`: pipeline de certificação aritmética de \[ħ\] (sanity, HS↔τ, outer, BK↔FK, EAC, KMS, Satake)
  - `scripts/run_constants.py`: orquestra \[c\] (ICC) e \[ħ\] (pipeline)
- `auditoria/`: manifests, checksums e logs de execuções
- `paper/tables/`: tabelas e extratos (CSV/JSON) prontos para inclusão no LaTeX
- `tests/`: testes unitários mínimos (verificações de API e deprecations)
- `.github/workflows/`: pipelines de CI (lint, testes, validação básica)

## Uso rápido

### 1) Pipeline principal (artigo)
- Executar:
  - `python scripts/run_pipeline.py --seed 123 --prec 1e-12 --out auditoria/run_YYYYMMDD`
- Artefatos esperados:
  - Relatórios JSON em `auditoria/...`
  - Tabelas em `paper/tables/...`
- Validar checksums:
  - `python scripts/validate_checksums.py auditoria/run_YYYYMMDD`

### 2) Constantes fundamentais adicionadas

- c via ICC (reproduz o valor definido do SI a partir de \[μ_0\] e \[ε_0\]):
  - `python scripts/constants/c_from_icc.py --prec 50 --outdir paper/tables`
  - Saídas:
    - `paper/tables/c_icc_report.json`
    - `paper/tables/c_icc_summary.csv`

- ħ — Pipeline de certificação aritmética (sanity elíptica, HS↔τ, outer canônico, BK↔FK, EAC, KMS, Satake):
  - `python scripts/constants/hbar_pipeline.py`
  - Saídas principais:
    - `paper/tables/hbar_summary.json`
    - `paper/tables/hbar_summary.csv`
    - `outer_curve.csv`, `outer_report.json`
    - `bkfk_outer_report.json`
    - `eac_determinants_report.txt`, `eac_determinants_final.csv`, `eac_determinants_sensitivity.csv`
    - `kms_pressure.png`, `kms_pressure_curve.csv`, `kms_pressure_report.txt`
    - `satake_functoriality_report.json`

- Orquestração (c + ħ):
  - `python scripts/run_constants.py --prec 50`

## Testes
- Executar:
  - `pytest -q`
- Opcional:
  - `flake8` / `ruff` / `mypy` se configurados

## Reprodutibilidade
- Seeds fixadas em manifests
- Precisão decimal e tolerâncias documentadas
- SHA-256 de inputs/outputs canônicos

## Como citar
Se usar estes resultados, cite:
- Autor. “Teoria de Tudo — Artigo.” Repositório GitHub, 2025.

BibTeX:
```
@misc{teoriadetudo2025,
  title        = {Teoria de Tudo — Artigo},
  author       = {Paulo Sobrenome},
  year         = {2025},
  howpublished = {\url{https://github.com/fisicapaulo/teoria_de_tudo_artigo}}
}
```

## Licença
MIT — veja o arquivo LICENSE.

## Contato
Abra uma issue neste repositório ou entre em contato via página do autor.
