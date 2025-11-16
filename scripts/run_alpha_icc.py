# -*- coding: utf-8 -*-
# Gera alpha_icc_report.json com parâmetros, resultados e checksum.
# Este script ilustra o pipeline de reprodutibilidade: seeds, precisão e auditoria.

import os
import uuid
import time
import platform
import random
import numpy as np
from decimal import Decimal
from scripts.io_utils import write_json_with_sha256
from scripts.precision import decimal_context, set_mpmath_dps, enable_strict_deprecations


def main():
    # Semeadura determinística básica
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)

    enable_strict_deprecations()
    set_mpmath_dps(100)

    # Parâmetros de demonstração
    gamma = Decimal("1")
    a = Decimal("1")  # placeholder; na prática vem do ajuste/derivação
    pi = Decimal("3.1415926535897932384626433832795028841971")

    with decimal_context(prec=100):
        c_curv = (a / (pi * pi)) * (gamma + (Decimal(1) / gamma))
        alpha_icc0 = Decimal("1")  # placeholder para termo volumétrico canônico
        alpha_inv = alpha_icc0 + c_curv

    report = {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "decimal_prec": 100,
        "seeds": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "random": 0,
            "numpy": 0,
        },
        "params": {
            "gamma": str(gamma),
            "a": str(a),
            "notes": "Demonstração de pipeline. Substitua por valores do artigo."
        },
        "results": {
            "alpha_inv": str(alpha_inv)
        },
        "conditioning": {
            "notes": "Placeholders; insira estimativas reais quando disponíveis."
        },
        "validation": {
            "double_vs_high_prec_diff": "N/A"
        }
    }

    os.makedirs("auditoria", exist_ok=True)
    path = os.path.join("auditoria", "alpha_icc_report.json")
    digest = write_json_with_sha256(path, report, insert_checksum_field=True)

    run_manifest = {
        "env": {},
        "deps": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "artifacts": [
            {
                "artifact_path": path,
                "sha256": digest,
                "generator_script": "scripts/run_alpha_icc.py",
            }
        ],
    }
    write_json_with_sha256(os.path.join("auditoria", "run_manifest.json"), run_manifest, insert_checksum_field=True)
    print(f"Relatório gerado: {path}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
