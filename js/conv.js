// ===================================================================
// conv.js - sequences of random variables for the convergence demos
// ===================================================================

function makeRng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gauss(rng) {                         // Box-Muller
  let u = 0; while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// typewriter: block k (k = 1, 2, ...) has k sub-intervals of length 1/k, swept left to right
function typewriterIndex(n) {                 // n = 1, 2, ...  ->  [k, j] with 0 <= j < k
  let k = 1, start = 1;
  while (start + k <= n) { start += k; k++; }
  return [k, n - start];
}

// each generator returns a function path(rng, N) -> Float64Array of X_1..X_N for ONE omega
const SEQS = {
  mean: {
    name: "sample mean of Uniform(0,1), minus 1/2",
    path(rng, N) { const x = new Float64Array(N); let s = 0; for (let n = 1; n <= N; n++) { s += rng() - 0.5; x[n - 1] = s / n; } return x; },
    modes: { as: true, p: true, l2: true, d: true }, eps: 0.02, ymax: 0.25,
    note: "Strong LLN: converges a.s., and also in L² (E[(X̄−½)²] = 1/(12n)) and in probability.",
  },
  bern1: {
    name: "independent X_n ~ Bernoulli(1/n)",
    path(rng, N) { const x = new Float64Array(N); for (let n = 1; n <= N; n++) x[n - 1] = rng() < 1 / n ? 1 : 0; return x; },
    modes: { as: false, p: true, l2: true, d: true }, eps: 0.5, ymax: 1.1,
    note: "P(X_n = 1) = 1/n → 0, so in probability and in L². But Σ 1/n = ∞ and the X_n are independent, so by the second Borel-Cantelli lemma X_n = 1 infinitely often with probability 1: NOT a.s.",
  },
  bern2: {
    name: "independent X_n ~ Bernoulli(1/n²)",
    path(rng, N) { const x = new Float64Array(N); for (let n = 1; n <= N; n++) x[n - 1] = rng() < 1 / (n * n) ? 1 : 0; return x; },
    modes: { as: true, p: true, l2: true, d: true }, eps: 0.5, ymax: 1.1,
    note: "Σ 1/n² < ∞, so by the first Borel-Cantelli lemma only finitely many X_n are 1: converges a.s. (and in P, L²).",
  },
  spike: {
    name: "X_n = n · 1{U < 1/n}, one U for all n",
    path(rng, N) { const u = rng(), x = new Float64Array(N); for (let n = 1; n <= N; n++) x[n - 1] = u < 1 / n ? n : 0; return x; },
    modes: { as: true, p: true, l2: false, d: true }, eps: 0.5, ymax: null,
    note: "For every U > 0, X_n = 0 as soon as n > 1/U: converges a.s. But E[X_n²] = n² · (1/n) = n → ∞ (even E[X_n] = 1): NOT in L².",
  },
  typewriter: {
    name: "typewriter: indicator of an interval sliding over [0, 1]",
    path(rng, N) {
      const u = rng(), x = new Float64Array(N);
      for (let n = 1; n <= N; n++) { const [k, j] = typewriterIndex(n); x[n - 1] = u >= j / k && u < (j + 1) / k ? 1 : 0; }
      return x;
    },
    modes: { as: false, p: true, l2: true, d: true }, eps: 0.5, ymax: 1.1,
    note: "The interval has length 1/k → 0, so P(X_n = 1) → 0: in probability and in L². But in each block the interval sweeps the whole [0,1], so every ω gets X_n = 1 once per block, forever: NOT a.s. (for no ω at all).",
  },
};

// simulate M paths up to N; returns curves vs n and a few paths for drawing
function simulateModes(key, M, N, seed) {
  const S = SEQS[key], rng = makeRng(seed);
  const exceed = new Float64Array(N), sq = new Float64Array(N), tailExceed = new Float64Array(N);
  const show = [];
  for (let m = 0; m < M; m++) {
    const x = S.path(rng, N);
    let lastBig = -1;                            // last index where |X_n| > eps
    for (let n = 0; n < N; n++) {
      const a = Math.abs(x[n]);
      if (a > S.eps) { exceed[n]++; lastBig = n; }
      sq[n] += x[n] * x[n];
    }
    // sup_{k >= n} |X_k| > eps  <=>  n <= lastBig
    for (let n = 0; n <= lastBig; n++) tailExceed[n]++;
    if (m < 25) show.push(x);
  }
  for (let n = 0; n < N; n++) { exceed[n] /= M; sq[n] /= M; tailExceed[n] /= M; }
  return { exceed, sq, tailExceed, show };
}

// Z_n = sqrt(n)(mean_n - 1/2)/sigma and Z_2n from the same path
function cltPairs(n, M, seed) {
  const rng = makeRng(seed), sig = Math.sqrt(1 / 12), a = new Float64Array(M), b = new Float64Array(M);
  for (let m = 0; m < M; m++) {
    let s = 0;
    for (let i = 1; i <= 2 * n; i++) { s += rng() - 0.5; if (i === n) a[m] = s / (Math.sqrt(n) * sig); }
    b[m] = s / (Math.sqrt(2 * n) * sig);
  }
  return { a, b };
}

// paths of the running mean for AR(1) with correlation rho (Var X = 1), or Cauchy
function runningMeans(kind, rho, N, M, seed) {
  const rng = makeRng(seed), paths = [], finals = new Float64Array(M);
  for (let m = 0; m < M; m++) {
    let x = kind === "cauchy" ? 0 : gauss(rng), s = 0;
    const p = m < 20 ? new Float64Array(N) : null;
    for (let n = 1; n <= N; n++) {
      if (kind === "cauchy") x = Math.tan(Math.PI * (rng() - 0.5));
      else if (n > 1) x = rho * x + Math.sqrt(1 - rho * rho) * gauss(rng);
      s += x;
      if (p) p[n - 1] = s / n;
    }
    finals[m] = s / N;
    if (p) paths.push(p);
  }
  return { paths, finals };
}

if (typeof module !== "undefined") module.exports = { SEQS, simulateModes, cltPairs, runningMeans, typewriterIndex, makeRng };
