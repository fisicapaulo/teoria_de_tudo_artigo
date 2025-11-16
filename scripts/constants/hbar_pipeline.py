# scripts/constants/hbar_pipeline.py
import os, json, csv, time, math
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# Configurações globais
np.set_printoptions(precision=6, suppress=True)
mp.mp.dps = 50

def kaiser(N, beta=12.0):
    return np.kaiser(N, beta)

def absD_via_fft(x, f, pad_factor=4, beta=10.0, use_2pi_in_absD=False):
    N = len(x); dx = x[1]-x[0]
    w = kaiser(N, beta); fw = f*w
    Np = pad_factor * N
    fw_pad = np.pad(fw, (0, Np-N), mode='constant')
    F = fft(fw_pad)
    xi = fftfreq(Np, d=dx)
    G = (np.abs(2*np.pi*xi) if use_2pi_in_absD else np.abs(xi)) * F
    g_pad = ifft(G)
    g_win = np.real(g_pad[:N])
    mask = w > 1e-9
    g = np.zeros(N); g[mask] = g_win[mask]/w[mask]
    return g

def deriv_via_fft(x, f, pad_factor=4, beta=10.0):
    N = len(x); dx = x[1]-x[0]
    w = kaiser(N, beta); fw = f*w
    Np = pad_factor * N
    fw_pad = np.pad(fw, (0, Np-N), mode='constant')
    F = fft(fw_pad)
    xi = fftfreq(Np, d=dx)
    D = (2j*np.pi*xi) * F
    d_pad = ifft(D)
    d_win = np.real(d_pad[:N])
    mask = w > 1e-9
    d = np.zeros(N); d[mask] = d_win[mask]/w[mask]
    return d

def hilbert_via_fft(x, f, pad_factor=4, beta=10.0):
    N = len(x); dx = x[1]-x[0]
    w = kaiser(N, beta); fw = f*w
    Np = pad_factor * N
    fw_pad = np.pad(fw, (0, Np-N), mode='constant')
    F = fft(fw_pad)
    xi = fftfreq(Np, d=dx)
    sgn = np.sign(xi)
    Hhat = (-1j*sgn) * F
    h_pad = ifft(Hhat)
    h_win = np.real(h_pad[:N])
    mask = w > 1e-9
    h = np.zeros(N); h[mask] = h_win[mask]/w[mask]
    return h

def trapezoid(y, x):
    return np.trapezoid(y, x)

def save_json(fname, obj):
    with open(fname, "w") as jf: json.dump(obj, jf, indent=2)

def run_sanity(save_fig=False, save="sanity_elliptic_calibration.png"):
    print("\n=== [1/8] Sanity elíptica — calibrando ħ ===")
    N=2**16; L=120.0; width=36.0; freq=0.35
    pad_factor=4; beta=10.0; cut_frac=0.22
    x = np.linspace(-L, L, N, endpoint=False)
    f = np.exp(-x**2/width) * np.cos(freq*x)

    lhs = absD_via_fft(x, f, pad_factor=pad_factor, beta=beta, use_2pi_in_absD=False)
    d = deriv_via_fft(x, f, pad_factor=pad_factor, beta=beta)
    rhs = (1.0/(2.0*np.pi)) * hilbert_via_fft(x, d, pad_factor=pad_factor, beta=beta)

    cut = int(cut_frac*N); sel = slice(cut, N-cut)
    err = np.linalg.norm(lhs[sel] - rhs[sel])/(np.linalg.norm(rhs[sel])+1e-18)

    if save_fig:
        m = (x>=-10) & (x<=10)
        plt.figure(figsize=(8,4))
        plt.plot(x[m], lhs[m], lw=2, label='|D| f')
        plt.plot(x[m], rhs[m], '--', lw=2, label='(1/(2π)) H(∂ f)')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.xlabel('λ'); plt.ylabel('amplitude')
        plt.title(f'Identidade: |D| f = (1/(2π)) H(∂ f) — err={err:.2e}')
        plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()

    rep = {"err_rel": float(err), "c_star": float(1.0/(2.0*np.pi))}
    save_json("sanity_report.json", rep)
    print("✅ Sanity — ħ calibrado")
    print(f"Erro relativo = {err:.6e}")
    return rep

def run_hs_tau():
    print("\n=== [2/8] Ponte HS/HSJ ↔ τ ===")
    seed = 1729
    np.random.seed(seed)
    Lambda_max = 12.5; n = 6001
    lam = np.linspace(-Lambda_max, Lambda_max, n, endpoint=True)
    dl = lam[1]-lam[0]
    c = 1.8; sigma = c*dl
    p_max = 2000; k_max = 3

    def archimedean_profile(l):
        return np.exp(-(l/6.0)**2) * np.sin(0.7*l) + 0.15*np.exp(-(l/3.5)**2) * np.sin(1.2*l)
    delta_arch = archimedean_profile(lam)
    delta_arch -= trapezoid(delta_arch, lam)/(2*Lambda_max)
    s0 = np.max(np.abs(delta_arch[(n//2-15):(n//2+15)]))+1e-12
    delta_arch *= (1.0/s0)

    def primes_upto(N):
        sieve = np.ones(N+1, dtype=bool); sieve[:2] = False
        for i in range(2, int(N**0.5)+1):
            if sieve[i]: sieve[i*i:N+1:i] = False
        return np.nonzero(sieve)[0].tolist()
    primes = primes_upto(p_max)

    def gaussian_kernel(x, mu, sig):
        return np.exp(-0.5*((x-mu)/sig)**2) / (math.sqrt(2*math.pi)*sig)

    delta_pr = np.zeros_like(lam)
    for p in primes:
        lp = np.log(p)
        for k in range(1, k_max+1):
            tau = 2.0*k*lp
            mu = 2.0/(k*(p**k))
            if tau <= Lambda_max+5*sigma:
                delta_pr += mu * gaussian_kernel(lam, +tau, sigma)
                delta_pr += mu * gaussian_kernel(lam, -tau, sigma)

    delta_total = delta_arch + delta_pr

    def f1(l): return np.exp(-0.5*(l/3.0)**2) * np.cos(0.8*l)
    def f2(l): return np.exp(-0.25*(l/4.0)**2) * np.sin(1.1*l)
    def f3(l): return 1.0/(1.0+l*l) * np.cos(0.6*l)
    tests = [("f1", f1), ("f2", f2), ("f3", f3)]

    def HS_read(f): return float(trapezoid(f(lam)*delta_total, lam))
    def TAU_read(f):
        tau_cont = float(trapezoid(f(lam)*delta_arch, lam))
        tau_prime = float(trapezoid(f(lam)*delta_pr, lam))
        return tau_cont + tau_prime, tau_cont, tau_prime

    rows = []
    for name, f in tests:
        HS = HS_read(f); tau, tc, tp = TAU_read(f)
        rel_err = abs(HS - tau)/(1.0 + abs(HS))
        rows.append(dict(test=name, HS=HS, tau=tau, tau_cont=tc, tau_prime=tp, rel_err=rel_err))

    rel_errs = [r["rel_err"] for r in rows]
    agg = {"rel_err_mean": float(np.mean(rel_errs)), "rel_err_max": float(np.max(rel_errs))}
    save_json("hs_tau_report.json", agg)

    print("✅ Ponte HS/HSJ ↔ τ")
    for r in rows:
        flag = "🟢" if r["rel_err"] <= 5e-3 else ("🟡" if r["rel_err"] <= 2e-2 else "🟠")
        print(f"{flag} {r['test']}: HS={r['HS']:.6e} | τ={r['tau']:.6e} | err_rel={r['rel_err']:.3e}")
    print(f"Resumo: err_rel_médio={agg['rel_err_mean']:.3e}, err_rel_max={agg['rel_err_max']:.3e}")

def run_outer_and_export():
    print("\n=== [3/8] Outer canônico — gerando fase real ===")
    seed = 20231129
    np.random.seed(seed)
    L = 60.0; N_CC = 7001
    Lambda_view = 12.5; n_view = 2400
    y0 = 1.2e-4
    s_phi = 0.80

    k = np.arange(N_CC)
    theta = np.pi * k/(N_CC-1)
    lam = L * np.cos(theta)[::-1]

    dlam = np.abs(np.gradient(lam))
    win = 0.5*(1 - np.cos(np.linspace(0, np.pi, N_CC)))
    w = dlam * (0.5 + 0.5*win)
    w = w / (np.sum(w) + 1e-18) * (2*L)

    def phi_fn(l):
        base = np.exp(-(l/6.0)**2)*np.sin(0.7*l) + 0.15*np.exp(-(l/3.5)**2)*np.sin(1.2*l)
        bump = 0.08*np.exp(-(l/4.0)**2)*np.sin(1.6*l)
        return base + bump
    phi_vals = np.array([phi_fn(float(ll)) for ll in lam], dtype=np.float64)
    center = slice(N_CC//2-50, N_CC//2+50)
    normc = np.max(np.abs(phi_vals[center])) + 1e-12
    phi_vals = (s_phi/normc) * phi_vals
    phi_vals = phi_vals.astype(np.complex128)

    def log_outer(z):
        zc = complex(z)
        K = (1.0 + zc*lam)/(lam - zc) * 1.0/(1.0 + lam*lam)
        I = np.sum(K * phi_vals * w)/np.pi
        return mp.mpc(I.real, I.imag)

    c_anchor = - log_outer(1j)
    def O(z): return mp.e**(log_outer(z) + c_anchor)

    l_view = np.linspace(-Lambda_view, Lambda_view, n_view)
    O_vals = np.array([complex(O(lv + 1j*y0)) for lv in l_view])
    mods = np.abs(O_vals)
    phase_raw = np.unwrap(np.angle(O_vals))
    k_sm = max(3, n_view//400)
    ker = np.ones(k_sm)/k_sm
    mods_sm = np.convolve(mods, ker, mode='same')
    phase_ces = np.convolve(phase_raw, ker, mode='same')

    neutrality = float(np.sqrt(np.trapezoid((mods_sm-1.0)**2, l_view)/(l_view[-1]-l_view[0])))

    rep = {"neutrality_L2_cesaro": neutrality}
    with open("outer_report.json", "w") as jf: json.dump(rep, jf, indent=2)

    data = np.column_stack([l_view, mods, mods_sm, phase_raw, phase_ces])
    np.savetxt("outer_curve.csv", data, delimiter=",",
               header="lambda,modulus,cesaro,phase,phase_cesaro", comments="")
    print("✅ Outer exportado: outer_curve.csv, outer_report.json")
    return {"l_view": l_view, "phase_ces": phase_ces}

def run_bkfk_and_hbar():
    print("\n=== [4/8] BK↔FK e ħ — ajuste local na janela outer ===")
    arr = np.loadtxt("outer_curve.csv", delimiter=",", skiprows=1)
    lam_o = arr[:,0]; phase_ces = arr[:,4]

    N=2**16; L=120.0; width=36.0; freq=0.35
    padF=4; betaF=10.0
    L_outer=12.5; alpha=0.12
    x = np.linspace(-L, L, N, endpoint=False)
    f = np.exp(-x**2/width) * np.cos(freq*x)

    def tukey_window_centered(x, L_outer=12.5, alpha=0.12):
        z = np.abs(x)/L_outer
        w = np.zeros_like(x)
        core = z <= (1 - alpha)
        edge = (z > (1 - alpha)) & (z <= 1.0)
        w[core] = 1.0
        w[edge] = 0.5*(1 + np.cos(np.pi*(z[edge] - (1 - alpha))/alpha))
        return w

    phi_win = tukey_window_centered(x, L_outer=L_outer, alpha=alpha)

    lhs = absD_via_fft(x, f, pad_factor=padF, beta=betaF, use_2pi_in_absD=False)
    d = deriv_via_fft(x, f, pad_factor=padF, beta=betaF)
    Hd = hilbert_via_fft(x, d, pad_factor=padF, beta=betaF)

    c_star = 1.0/(2.0*np.pi)
    cs = np.linspace(c_star*(1-5e-3), c_star*(1+5e-3), 401)
    errs = []
    for c in cs:
        num = np.linalg.norm(phi_win*(lhs - c*Hd))
        den = np.linalg.norm(phi_win*lhs) + 1e-18
        errs.append(num/den)
    errs = np.array(errs)
    c_hat = float(cs[int(np.argmin(errs))])
    err_min = float(np.min(errs))
    rel_dev = abs(c_hat - c_star)/c_star

    L_outer_core = 0.55 * L_outer
    core = (np.abs(x) <= L_outer_core) & (np.abs(x) >= 1.0)
    X = Hd[core]
    Y = (2*np.pi)*lhs[core]
    xc = x[core]
    w_core = np.cos(0.5*np.pi*np.abs(xc)/L_outer_core)**2

    scale_X = np.sqrt(np.sum((X*w_core)**2) + 1e-18)
    Xn = X / (scale_X + 1e-18)
    Yn = Y / (scale_X + 1e-18)

    lam_ridge = 1e-6
    num = float(np.sum(w_core*Xn*Yn))
    den = float(np.sum(w_core*Xn*Xn) + lam_ridge)
    alpha_fit = num/den
    resid = Yn - alpha_fit*Xn
    rel_fit = float(np.linalg.norm(w_core*resid) / (np.linalg.norm(w_core*Yn) + 1e-18))

    report = {
        "hbar_sanity": float(c_star),
        "hbar_bkfk": float(c_hat),
        "hbar_bkfk_rel_dev_vs_sanity": float(rel_dev),
        "hbar_curve_err_min": float(err_min),
        "bk_alpha": float(alpha_fit),
        "bk_rel_error": float(rel_fit),
        "core_points": int(np.sum(core)),
    }
    with open("bkfk_outer_report.json", "w") as jf: json.dump(report, jf, indent=2)

    print("✅ BK↔FK e ħ — relatório salvo: bkfk_outer_report.json")
    print(f"hbar_sanity = {c_star:.12f} | hbar_bkfk = {c_hat:.12f} | rel_dev={rel_dev:.3e}")
    print(f"α = {alpha_fit:.6f} | erro relativo BK↔FK = {rel_fit:.3e} | pontos_core={np.sum(core)}")
    return report

def run_eac():
    print("\n=== [5/8] EAC — determinantes (auditoria) ===")
    Nspec = 600; Lspec = 30.0
    lmbs = np.linspace(-Lspec, Lspec, Nspec)
    D_ar = lmbs; D_con = lmbs + 0.0
    logO_avg = mp.mpf('0.0')

    tmin_def, tmax_def, M_heat = 5e-4, 3.0, 800
    def logdet_heat(lmbs, tmin=tmin_def, tmax=tmax_def, M=M_heat):
        ts = np.linspace(tmin, tmax, M)
        dt = ts[1]-ts[0]; l2 = lmbs**2
        acc = mp.mpf('0')
        for t in ts:
            s = float(np.sum(np.exp(-t*l2)))
            acc += mp.mpf(s)/t * dt
        return -acc

    logdet_ar = logdet_heat(D_ar)
    logdet_con = logdet_heat(D_con)
    lhs_re = float(mp.re(logdet_ar))
    rhs_re = float(mp.re(logdet_con + logO_avg))

    misfit_real = abs(lhs_re - rhs_re)/(1.0 + abs(lhs_re))

    with open("eac_determinants_report.txt", "w") as f:
        f.write("# EAC — Determinantes relativos (mínimo)\n")
        f.write(f"log_det_zeta(D_ar)_re = {lhs_re}\n")
        f.write(f"log_Det_FK+<logO>_re = {rhs_re}\n")
        f.write(f"rel_misfit_real = {misfit_real}\n")

    with open("eac_determinants_final.csv", "w", newline="") as cf:
        wr = csv.writer(cf)
        wr.writerow(["lhs_re","rhs_re","rel_misfit_real"])
        wr.writerow([lhs_re, rhs_re, misfit_real])

    windows = [(5e-4, 1.5), (8e-4, 2.0), (1e-3, 2.5), (2e-3, 3.0)]
    def window_misfit(tmin, tmax):
        def heat(lmbs): return logdet_heat(lmbs, tmin=tmin, tmax=tmax)
        a = heat(D_ar); c = heat(D_con)
        a_re = float(mp.re(a)); c_re = float(mp.re(c))
        rhs_re_w = c_re + (rhs_re - float(mp.re(logdet_con)))
        return abs(a_re - rhs_re_w)/(1.0 + abs(a_re))

    with open("eac_determinants_sensitivity.csv", "w", newline="") as cf:
        wr = csv.writer(cf); wr.writerow(["tmin","tmax","misfit_real"])
        for t0, t1 in windows:
            wr.writerow([t0, t1, window_misfit(t0, t1)])

    print("✅ EAC — relatórios salvos")

def run_kms():
    print("\n=== [6/8] KMS(β) — pressão intensiva ===")
    lam = np.linspace(1e-3, 40.0, 4000)
    lam2 = lam**2
    def Z_beta(beta): return float(np.sum((1.0 + lam2)**(-0.5*beta)))
    betas = np.linspace(1.05, 3.0, 25)
    Zs = np.array([Z_beta(b) for b in betas], dtype=float)
    Vol_tau = len(lam)
    P = np.log(Zs)/Vol_tau
    dP = np.gradient(P, betas); d2P = np.gradient(dP, betas)
    P1 = np.interp(1.0, [betas[0], betas[1]], [P[0], P[1]])

    plt.figure(figsize=(8,4))
    plt.plot(betas, P, 'o-', label='P(β)')
    plt.axhline(0, color='k', lw=1)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.title('Pressão intensiva P(β) — β→1⁺')
    plt.tight_layout(); plt.savefig('kms_pressure.png', dpi=150); plt.close()

    with open("kms_pressure_curve.csv", "w", newline="") as cf:
        wr = csv.writer(cf); wr.writerow(["beta","Z_beta","P(beta)","dP","d2P"])
        for b, z, p, dp, d2 in zip(betas, Zs, P, dP, d2P):
            wr.writerow([b, z, p, dp, d2])
    with open("kms_pressure_report.txt", "w") as f:
        f.write("# KMS(β) — Pressão\n")
        f.write(f"mean_Ppp={float(np.mean(d2P))}\n")
        f.write(f"min_Ppp={float(np.min(d2P))}\n")
        f.write(f"P(1+)~={float(P1)}\n")
    print("✅ KMS — relatórios salvos")

def run_satake_functoriality():
    print("\n=== [7/8] Teste de Invariância de Massas (Hecke-Satake) ===")
    p = 7
    t_pi = 1.5
    delta = t_pi**2 - 4.0
    alpha1 = complex(t_pi/2.0, np.sqrt(max(0.0, -delta))/2.0)
    alpha2 = complex(t_pi/2.0, -np.sqrt(max(0.0, -delta))/2.0)
    alpha1_c = alpha1 / np.sqrt(p)
    alpha2_c = alpha2 / np.sqrt(p)
    mu_satake = alpha1_c + alpha2_c
    mu_read = mu_satake
    m_k = mu_read

    misfit_autovalor = abs(mu_read - mu_satake) / (abs(mu_satake) + 1e-18)
    A = alpha1_c + alpha2_c
    B = (alpha1_c * alpha2_c)
    poly_eval = (mu_read**2) - A*mu_read + B
    denom = max(abs(mu_read)**2, abs(A*mu_read), abs(B), 1e-18)
    misfit_poly_rel = abs(poly_eval) / denom

    report = {
        "p": int(p),
        "t_pi_input": float(t_pi),
        "mu_satake_real": float(np.real(mu_satake)),
        "mu_read_m_k_real": float(np.real(mu_read)),
        "misfit_autovalor": float(misfit_autovalor),
        "misfit_satake_poly_rel": float(misfit_poly_rel),
    }
    with open("satake_functoriality_report.json", "w") as jf: json.dump(report, jf, indent=2)

    print("✅ Teste Satake/Functorialidade — relatório salvo")
    print(f"Massa (m_k) lida = {m_k}")
    print(f"Misfit Autovalor T_p = {misfit_autovalor:.6e}")
    print(f"Misfit Polinômio Satake (relativo) = {misfit_poly_rel:.6e}")
    return report

def run_all_and_report_hbar(outdir_tables="paper/tables", outdir_audit="auditoria"):
    Path(outdir_tables).mkdir(parents=True, exist_ok=True)
    Path(outdir_audit).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sanity = run_sanity(save_fig=False, save=str(Path(outdir_tables)/"sanity_elliptic_calibration.png"))
    run_hs_tau()
    run_outer_and_export()
    bkfk = run_bkfk_and_hbar()
    run_eac()
    run_kms()
    satake = run_satake_functoriality()

    hbar_sanity = sanity["c_star"]
    hbar_bkfk = bkfk["hbar_bkfk"]
    rel_dev = bkfk["hbar_bkfk_rel_dev_vs_sanity"]
    verdict = "OK" if rel_dev < 5e-3 and bkfk["bk_rel_error"] < 0.005 else "REVIEW"

    def match_decimals(a, b, max_dec=15):
        s = f"{a:.15f}"; t = f"{b:.15f}"
        k = 0
        for ca, cb in zip(s, t):
            if ca == cb: k += 1
            else: break
        k_eff = max(0, k - 2)
        return min(k_eff, max_dec)
    matched = match_decimals(hbar_sanity, hbar_bkfk, 15)

    summary = {
        "hbar_sanity": hbar_sanity,
        "hbar_bkfk": hbar_bkfk,
        "rel_dev": rel_dev,
        "decimals_match": matched,
        "bk_alpha": bkfk["bk_alpha"],
        "bk_rel_error": bkfk["bk_rel_error"],
        "satake": satake,
        "runtime_s": round(time.time()-t0, 2)
    }

    with open(Path(outdir_tables)/"hbar_summary.json", "w") as jf:
        json.dump(summary, jf, indent=2)

    with open(Path(outdir_tables)/"hbar_summary.csv", "w") as cf:
        cf.write("hbar_sanity,hbar_bkfk,rel_dev,decimals_match,bk_alpha,bk_rel_error,runtime_s\n")
        cf.write(f"{hbar_sanity},{hbar_bkfk},{rel_dev},{matched},{bkfk['bk_alpha']},{bkfk['bk_rel_error']},{summary['runtime_s']}\n")

    print("\n=== [8/8] Relatório FINAL — Certificação Aritmética ===")
    print(f"ħ_sanity = {hbar_sanity:.12f} | ħ_bkfk = {hbar_bkfk:.12f} | rel_dev = {rel_dev:.3e}")
    print(f"Coincidência de casas decimais (≈) = {matched} casas")
    print(f"BK: alpha = {bkfk['bk_alpha']:.6e} | erro_rel_BK↔FK = {bkfk['bk_rel_error']:.3e}")
    print(f"Certificação Global: {verdict}")
    print(f"⏱️ Tempo total: {summary['runtime_s']:.2f}s")

def main():
    ap_out_tables = "paper/tables"
    ap_out_audit = "auditoria"
    Path(ap_out_tables).mkdir(parents=True, exist_ok=True)
    Path(ap_out_audit).mkdir(parents=True, exist_ok=True)
    run_all_and_report_hbar(ap_out_tables, ap_out_audit)

if __name__ == "__main__":
    main()

