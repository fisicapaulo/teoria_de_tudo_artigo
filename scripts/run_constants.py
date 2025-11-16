# scripts/run_constants.py
import argparse
from pathlib import Path
import subprocess
import sys

def run_py(mod_path, args=None):
    cmd = [sys.executable, mod_path] + (args or [])
    r = subprocess.run(cmd, check=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser(description="Executa constantes: c (ICC) e ħ (pipeline).")
    ap.add_argument("--prec", type=int, default=50, help="Precisão para c (Decimal)")
    ap.add_argument("--out", type=str, default="paper/tables", help="Diretório de saída para relatórios rápidos")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # c (ICC)
    run_py("scripts/constants/c_from_icc.py", ["--prec", str(args.prec), "--outdir", args.out])

    # ħ (pipeline completo)
    run_py("scripts/constants/hbar_pipeline.py")

    print("[run_constants] Finalizado com sucesso.")

if __name__ == "__main__":
    main()
