# Teoria de Tudo — Artigo

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
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1
   - Linux/macOS:
     - python3 -m venv .venv
     - source .venv/bin/activate
2) Instalar dependências:
   - pip install -r requirements.txt

## Estrutura
- /scripts: scripts executáveis (Python) para cálculos, geração de relatórios e auditoria
- /auditoria: manifests, checksums e logs de execuções
- /paper/tables: tabelas e extratos (CSV/JSON) prontos para inclusão no LaTeX
- /tests: testes unitários mínimos (verificações de API e deprecations)
- /.github/workflows: pipelines de CI (lint, testes, validação básica)

## Uso rápido (end-to-end)
- Rodar pipeline principal:
  - python scripts/run_pipeline.py --seed 123 --prec 1e-12 --out auditoria/run_YYYYMMDD
- Artefatos esperados:
  - Relatórios JSON em `auditoria/...`
  - Tabelas em `paper/tables/...`
- Validar checksums:
  - python scripts/validate_checksums.py auditoria/run_YYYYMMDD

## Testes
- pytest -q
- Opcional: flake8 / ruff / mypy se configurados

## Reprodutibilidade
- Seeds fixadas em manifests
- Precisão decimal e tolerâncias documentadas
- SHA-256 de inputs/outputs canônicos

## Como citar
Se usar estes resultados, cite:
- Autor. “Teoria de Tudo — Artigo.” Repositório GitHub, 2025.
  - BibTeX:
    - @misc{teoriadetudo2025,
        title = {Teoria de Tudo — Artigo},
        author = {Paulo Sobrenome},
        year = {2025},
        howpublished = {\url{https://github.com/fisicapaulo/teoria_de_tudo_artigo}}
      }

## Licença
MIT — veja o arquivo LICENSE.

## Contato
Abra uma issue neste repositório ou entre em contato via página do autor.
