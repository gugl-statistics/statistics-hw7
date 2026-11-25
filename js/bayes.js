// ===================================================================
// bayes.js - Bayes' theorem, Beta posteriors, naive Bayes
// ===================================================================

// ---------- 1. one application of Bayes' theorem for a yes/no hypothesis
function bayesUpdate(prior, pDataIfTrue, pDataIfFalse) {
  const evidence = pDataIfTrue * prior + pDataIfFalse * (1 - prior);   // law of total probability
  return (pDataIfTrue * prior) / evidence;
}

// ---------- 2. Beta distribution
const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
const lbeta = (a, b) => lgamma(a) + lgamma(b) - lgamma(a + b);
function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b));
}
function betacf(a, b, x) {                      // continued fraction for the incomplete beta function
  const tiny = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  d = 1 / (Math.abs(d) > tiny ? d : tiny);
  let h = d;
  for (let m = 1; m < 1000; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; d = 1 / (Math.abs(d) > tiny ? d : tiny);
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; d = 1 / (Math.abs(d) > tiny ? d : tiny);
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}
function betaCdf(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = -lbeta(a, b) + a * Math.log(x) + b * Math.log(1 - x);
  if (x < (a + 1) / (a + b + 2)) return (Math.exp(lbt) * betacf(a, b, x)) / a;
  return 1 - (Math.exp(lbt) * betacf(b, a, 1 - x)) / b;
}
function betaPpf(q, a, b) {
  let lo = 0, hi = 1;
  for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (betaCdf(m, a, b) < q) lo = m; else hi = m; }
  return (lo + hi) / 2;
}
// binomial likelihood of p for k successes out of n, scaled so that it integrates to 1 over [0,1]
// (that scaled version is exactly the Beta(k+1, n-k+1) density)
const likelihoodScaled = (p, k, n) => betaPdf(p, k + 1, n - k + 1);

// ---------- 3. multinomial naive Bayes (model from code/bayes.py)
const tokenize = (s) => (s.toLowerCase().match(/[a-z0-9£$']+/g) || []);
function classify(text, priorSpam, alpha = 1) {
  const V = NB.V, ts = NB.spam_words, th = NB.ham_words;
  const terms = [];
  let lo = Math.log(priorSpam) - Math.log(1 - priorSpam);
  const prior = lo;
  for (const w of tokenize(text)) {
    const c = NB.vocab[w];
    if (!c) { terms.push({ w, known: false }); continue; }
    const ls = (c[0] + alpha) / (ts + alpha * V), lh = (c[1] + alpha) / (th + alpha * V);
    const llr = Math.log(ls) - Math.log(lh);
    lo += llr;
    terms.push({ w, known: true, spam: c[0], ham: c[1], llr });
  }
  const p = lo > 30 ? 1 : lo < -30 ? 0 : 1 / (1 + Math.exp(-lo));
  return { logOdds: lo, prior, p, terms };
}

if (typeof module !== "undefined") module.exports = { bayesUpdate, betaPdf, betaCdf, betaPpf, lgamma };
