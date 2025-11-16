# -*- coding: utf-8 -*-
# Calcula m_H a partir de M*_ICC e v_fisico/(2π), registra relatório e checksum.

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
    # Semeadura determinística
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)

    enable_strict_deprecations()
    set_mpmath_dps(100)

    # Parâmetros do artigo (ajuste conforme necessário)
    pi = Decimal("3.1415926535897932384626433832795028841971")
    gamma = Decimal("1")  # janela simétrica UV/IR
    v_fisico_GeV = Decimal("246.21965")  # exemplo; ajuste com valor oficial

    with decimal_context(prec=100):
        term = Decimal(1) + (Decimal(2) * gamma * gamma) / (pi ** 4)
        M_star = (Decimal(4) * pi * gamma.sqrt() / Decimal(3)) * term.sqrt()
        m_H = M_star * (v_fisico_GeV / (Decimal(2) * pi))

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
            "v_fisico_GeV": str(v_fisico_GeV),
        },
        "results": {
            "M_star": str(M_star),
            "m_H_GeV": str(m_H)
        },
        "validation": {
            "notes": "Compare com valor experimental e com alta precisão se necessário."
        }
    }

    os.makedirs("auditoria", exist_ok=True)
    path = os.path.join("auditoria", "spectral_mass_report.json")
    digest = write_json_with_sha256(path, report, insert_checksum_field=True)

    audit_manifest = {
        "run_id": report["run_id"],
        "artifacts": [
            {
                "artifact_path": path,
                "sha256": digest,
                "generator_script": "scripts/run_spectral_mass.py",
            }
        ]
    }
    write_json_with_sha256(os.path.join("auditoria", "audit_manifest.json"), audit_manifest, insert_checksum_field=True)
    print(f"Relatório gerado: {path}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
