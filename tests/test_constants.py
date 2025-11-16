# tests/test_constants.py
import json
from pathlib import Path
import subprocess, sys

def test_c_icc_runs(tmp_path: Path):
    outdir = tmp_path / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "scripts/constants/c_from_icc.py", "--prec", "50", "--outdir", str(outdir)], check=True)
    js = json.loads((outdir/"c_icc_report.json").read_text(encoding="utf-8"))
    assert "c_deduced" in js and "c_defined" in js
    assert js["status"] == "OK"

def test_hbar_pipeline_runs(tmp_path: Path, monkeypatch):
    # Rodar em diretório temporário para gerar artefatos lá
    cwd = Path.cwd()
    try:
        # Copiamos a execução a partir da raiz (gera arquivos na raiz),
        # mas testamos apenas que não quebra e gera summary.
        subprocess.run([sys.executable, "scripts/constants/hbar_pipeline.py"], check=True)
        assert Path("paper/tables/hbar_summary.json").exists()
    finally:
        pass
