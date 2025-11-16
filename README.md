# Teoria de Tudo — Artigo

Repositório de reprodução e auditoria do artigo:
- Fechamento de α^{-1} por ICC0 + ICC1
- Certificação espectral de m_H via fator geométrico-universal

Este repositório segue um protocolo de reprodutibilidade forte: manifests imutáveis, seeds fixadas, precisão decimal explícita e relatórios JSON canônicos com checksums SHA-256.

## Estrutura
- /scripts: scripts executáveis (Python) para cálculos, geração de relatórios e auditoria.
- /auditoria: manifests, checksums e logs de execuções.
- /paper/tables: tabelas e extratos (CSV/JSON) prontos para inclusão no LaTeX.
- /tests: testes unitários mínimos (verificações de API e deprecations).
- /.github/workflows: pipelines de CI (lint, testes, validação básica).

## Como usar
1) Crie e ative um ambiente virtual.
2) Instale dependências com `pip install -r requirements.txt`.
3) Rode os scripts em /scripts conforme instruções do README e dos headers dos próprios arquivos.

## Licença
Veja LICENSE.

## Contato
Abra uma issue neste repositório ou entre em contato via página do autor.
