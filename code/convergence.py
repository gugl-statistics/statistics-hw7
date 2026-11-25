"""
Homework 7 (part A) - Modes of convergence: simulations for the counterexamples, the LLN / CLT,
the sqrt(n) scaling and what happens with dependence or without a variance.
Python 3 + numpy.  Run:  python3 convergence.py
"""
import json, math
import numpy as np

rng = np.random.default_rng(2026)
out = {}

# ------------------------------------------------------------------ 1. the counterexamples
# all of them are sequences X_n(omega) built on the same probability space
M = 20000                     # number of omegas (sample paths)
NMAX = 2000

def typewriter(u, n):
    """sliding indicator: block k (k = 1, 2, ...) has k sub-intervals of length 1/k, swept left to right;
    X_n = 1 if u is in the current sub-interval"""
    k, start = 1, 1
    while start + k <= n:
        start += k; k += 1
    j = n - start
    return ((u >= j / k) & (u < (j + 1) / k)).astype(float)

def check(name, gen, ns=(10, 100, 500, 1000), eps=0.5):
    """estimates P(|X_n| > eps), E[X_n^2], and P(sup_{k >= n} |X_k| > eps) (a.s. criterion) up to NMAX"""
    X = gen()                                           # shape (M, NMAX), column n-1 = X_n
    absX = np.abs(X)
    # tail sup: sup over k >= n (within the simulated horizon)
    tail = np.maximum.accumulate(absX[:, ::-1], axis=1)[:, ::-1]
    rows = []
    for n in ns:
        rows.append(dict(n=n, p_exceed=float(np.mean(absX[:, n - 1] > eps)), mean_sq=float(np.mean(X[:, n - 1] ** 2)),
                         p_tail=float(np.mean(tail[:, n - 1] > eps))))
    print(f"\n{name}")
    for r in rows:
        print(f"  n={r['n']:5d}  P(|X_n|>{eps})={r['p_exceed']:.4f}  E[X_n^2]={r['mean_sq']:.4f}  P(sup_k>=n |X_k|>{eps}, k<= {NMAX})={r['p_tail']:.4f}")
    return rows

U = rng.random(M)
# (a) Bernoulli(1/n), independent: in P and L2, not a.s. (2nd Borel-Cantelli: sum 1/n = inf)
out["bern_1_n"] = check("independent Bernoulli(1/n)", lambda: (rng.random((M, NMAX)) < 1 / np.arange(1, NMAX + 1)).astype(float))
# (b) Bernoulli(1/n^2), independent: a.s. too (1st Borel-Cantelli)
out["bern_1_n2"] = check("independent Bernoulli(1/n^2)", lambda: (rng.random((M, NMAX)) < 1 / np.arange(1, NMAX + 1) ** 2).astype(float))
# (c) X_n = n * 1{U < 1/n}: a.s. (and in P) but not in L1/L2 (E X_n^2 = n)
out["spike"] = check("X_n = n 1{U<1/n}", lambda: np.arange(1, NMAX + 1) * (U[:, None] < 1 / np.arange(1, NMAX + 1)))
# (d) typewriter: in P and L2, not a.s. (every omega is hit once in every block)
out["typewriter"] = check("typewriter", lambda: np.stack([typewriter(U, n) for n in range(1, NMAX + 1)], axis=1))
# number of times X_n = 1 for each omega up to NMAX
hits_tw = np.stack([typewriter(U[:2000], n) for n in range(1, NMAX + 1)], axis=1).sum(axis=1)
print("  typewriter: number of n<=2000 with X_n = 1, per omega: min", hits_tw.min(), "max", hits_tw.max())
out["typewriter_hits"] = [int(hits_tw.min()), int(hits_tw.max())]

# (e) in distribution, not in probability: X_n = -Z
Z = rng.standard_normal(M)
out["minusZ"] = dict(p_far=float(np.mean(np.abs(-Z - Z) > 0.5)), exact=float(math.erfc(0.25 / math.sqrt(2))))
print(f"\nX_n = -Z: same law as Z, but P(|X_n - Z| > 0.5) = P(|Z| > 0.25) = {out['minusZ']['exact']:.4f} (sim {out['minusZ']['p_far']:.4f}) for every n")

# ------------------------------------------------------------------ 2. LLN: the sample mean in every mode
# X_i ~ Uniform(0,1): mean 1/2, var 1/12
print("\nLLN for the mean of Uniform(0,1)")
M2, N2 = 5000, 10000
Xs = rng.random((M2, N2))
means = np.cumsum(Xs, axis=1) / np.arange(1, N2 + 1)
dev = np.abs(means - 0.5)
tail = np.maximum.accumulate(dev[:, ::-1], axis=1)[:, ::-1]
lln = []
for n in [10, 100, 1000, 10000]:
    r = dict(n=n, p_exceed=float(np.mean(dev[:, n - 1] > 0.02)), mse=float(np.mean(dev[:, n - 1] ** 2)), mse_th=1 / 12 / n,
             p_tail=float(np.mean(tail[:, n - 1] > 0.02)), cheb=min(1, 1 / 12 / n / 0.02 ** 2))
    lln.append(r)
    print(f"  n={n:6d}  P(|mean-1/2|>0.02)={r['p_exceed']:.4f} (Chebyshev <= {r['cheb']:.4f})  E[(mean-1/2)^2]={r['mse']:.3e} (theory {r['mse_th']:.3e})  P(sup_k>=n ...)={r['p_tail']:.4f}")
out["lln"] = lln

# ------------------------------------------------------------------ 3. CLT is only in distribution
# Z_n = sqrt(n) (mean_n - mu) / sigma ; corr(Z_n, Z_2n) = 1/sqrt(2) for every n
print("\nCLT: Z_n and Z_2n")
clt = []
sig = math.sqrt(1 / 12)
for n in [10, 100, 1000, 5000]:
    Zn = math.sqrt(n) * (means[:, n - 1] - 0.5) / sig
    Z2n = math.sqrt(2 * n) * (means[:, 2 * n - 1] - 0.5) / sig
    c = float(np.corrcoef(Zn, Z2n)[0, 1])
    far = float(np.mean(np.abs(Zn - Z2n) > 0.5))
    # P(|Z_n - Z_2n| > 0.5) -> P(|N(0, 2 - sqrt 2)| > 0.5)
    far_th = math.erfc(0.5 / math.sqrt(2 * (2 - math.sqrt(2))))
    clt.append(dict(n=n, corr=c, p_far=far, p_far_th=far_th, var_zn=float(Zn.var())))
    print(f"  n={n:5d}  corr(Z_n, Z_2n)={c:.4f} (1/sqrt2 = 0.7071)  P(|Z_n - Z_2n|>0.5)={far:.4f} (limit {far_th:.4f})  Var(Z_n)={Zn.var():.4f}")
out["clt"] = clt

# ------------------------------------------------------------------ 4. sqrt(n): S_n / n^alpha
print("\nVar(S_n / n^alpha) for Uniform(0,1) (sigma^2 = 1/12)")
sc = []
S = np.cumsum(Xs - 0.5, axis=1)
for a in [0.25, 0.5, 0.75, 1.0]:
    row = dict(alpha=a, var=[float(np.var(S[:, n - 1] / n ** a)) for n in [10, 100, 1000, 10000]])
    sc.append(row)
    print(f"  alpha={a}: " + "  ".join(f"{v:.3e}" for v in row["var"]))
out["scaling"] = sc

# ------------------------------------------------------------------ 5. dependence: AR(1) with correlation rho
print("\nAR(1): n Var(mean) vs sigma^2 (1+rho)/(1-rho)")
ar = []
for rho in [0.0, 0.5, 0.9, -0.5]:
    n = 2000; m = 4000
    e = rng.standard_normal((m, n)) * math.sqrt(1 - rho ** 2)       # stationary with Var(X) = 1
    x = np.empty((m, n)); x[:, 0] = rng.standard_normal(m)
    for t in range(1, n): x[:, t] = rho * x[:, t - 1] + e[:, t]
    v = n * x.mean(axis=1).var()
    th = (1 + rho) / (1 - rho)
    ar.append(dict(rho=rho, n_var=float(v), theory=th, ess=n / th))
    print(f"  rho={rho:5}: n Var(mean) = {v:.3f}  (theory {th:.3f}, iid would be 1)  effective sample size {n / th:.0f} out of {n}")
out["ar1"] = ar

# ------------------------------------------------------------------ 6. no variance: Cauchy
print("\nCauchy: the sample mean does not settle")
C = rng.standard_cauchy((4000, 10000))
cm = np.cumsum(C, axis=1) / np.arange(1, 10001)
cau = []
for n in [10, 100, 1000, 10000]:
    q = np.quantile(cm[:, n - 1], [0.25, 0.75])
    cau.append(dict(n=n, iqr=float(q[1] - q[0]), p_far=float(np.mean(np.abs(cm[:, n - 1]) > 1))))
    print(f"  n={n:6d}  IQR of the mean = {q[1] - q[0]:.3f} (Cauchy: 2.000)  P(|mean| > 1) = {np.mean(np.abs(cm[:, n - 1]) > 1):.3f} (exact 0.5)")
out["cauchy"] = cau

json.dump(out, open("results_convergence.json", "w"), indent=1)
