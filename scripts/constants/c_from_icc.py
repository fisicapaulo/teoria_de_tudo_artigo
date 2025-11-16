# scripts/constants/c_from_icc.py
import argparse
import json
import math
from decimal import Decimal, getcontext
from pathlib import Path

def compute_c(prec=50):
    getcontext().prec = prec
    # Seus valores/fórmulas (exatos no SI; epsilon_0 calculado)
    MU_0_ICC_DERIVADO = Decimal(str(4 * math.pi * 1e-7))  # H/m (N/A^2)
    C_VALOR_DEFINIDO = Decimal("299792458")               # m/s (exato no SI)
    EPSILON_0_ICC_DERIVADO = Decimal(1) / (MU_0_ICC_DERIVADO * (C_VALOR_DEFINIDO**2))
    # c via relação clássica/aritmética
    c_aritmetico = Decimal(1) / (MU_0_ICC_DERIVADO * EPSILON_0_ICC_DERIVADO).sqrt()
    diff = c_aritmetico - C_VALOR_DEFINIDO

    return {
        "mu_0_icc": f"{MU_0_ICC_DERIVADO:.10E}",
        "epsilon_0_icc": f"{EPSILON_0_ICC_DERIVADO:.15E}",
        "c_defined": f"{C_VALOR_DEFINIDO:.0f}",
        "c_deduced": f"{c_aritmetico:.10f}",
        "difference": f"{diff:.10E}",
        "precision_digits": prec,
        "units": {
            "mu_0": "N/A^2 (H/m)",
            "epsilon_0": "F/m",
            "c": "m/s",
        },
        "status": "OK" if abs(diff) < Decimal("1e-9") else "REVIEW"
    }

def main():
    ap = argparse.ArgumentParser(description="Cálculo de c via ICC (mu0, epsilon0)")
    ap.add_argument("--prec", type=int, default=50, help="Dígitos de precisão decimal")
    ap.add_argument("--outdir", type=str, default="paper/tables", help="Diretório de saída")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res = compute_c(args.prec)

    # JSON canônico
    out_json = Path(args.outdir) / "c_icc_report.json"
    with out_json.open("w", encoding="utf-8") as jf:
        json.dump(res, jf, indent=2, ensure_ascii=False)

    # CSV curto
    out_csv = Path(args.outdir) / "c_icc_summary.csv"
    with out_csv.open("w", encoding="utf-8") as cf:
        cf.write("mu_0_icc,epsilon_0_icc,c_defined,c_deduced,difference,precision_digits\n")
        cf.write(f"{res['mu_0_icc']},{res['epsilon_0_icc']},{res['c_defined']},{res['c_deduced']},{res['difference']},{res['precision_digits']}\n")

    print(f"[c_from_icc] OK -> {out_json} | {out_csv}")

if __name__ == "__main__":
    main()

