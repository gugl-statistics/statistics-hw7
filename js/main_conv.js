// ===================================================================
// main_conv.js - page logic for part A (modes of convergence)
// ===================================================================
const $ = (id) => document.getElementById(id);
const f4 = (x) => (x === 0 ? "0" : x < 1e-3 ? x.toExponential(2) : x.toFixed(4));
function seg(el, onChange) {
  const btns = [...el.querySelectorAll("button")];
  btns.forEach((b) => (b.onclick = () => { btns.forEach((x) => x.classList.toggle("on", x === b)); onChange && onChange(b.dataset.v); }));
  return { get: () => el.querySelector("button.on").dataset.v };
}
const yes = (b) => (b ? '<span class="ok">yes</span>' : '<span class="bad">no</span>');

// =================================================== section 3: counterexample table (Python)
(function cxTable() {
  const rows = [["bern_1_n", "Bernoulli(1/n), independent", "yes", "yes", "no"], ["bern_1_n2", "Bernoulli(1/n²), independent", "yes", "yes", "yes"],
    ["typewriter", "typewriter", "yes", "yes", "no"], ["spike", "n·1{U < 1/n}", "yes", "no", "yes"]];
  let t = `<tr><th>sequence</th><th class="num">n</th><th class="num">P(|X<sub>n</sub>| &gt; ε)</th><th class="num">E[X<sub>n</sub>²]</th><th class="num">P(sup<sub>k≥n</sub>|X<sub>k</sub>| &gt; ε)</th><th>in P</th><th>in L²</th><th>a.s.</th></tr>`;
  for (const [k, name, p, l2, as] of rows) {
    PY[k].forEach((r, i) => {
      t += `<tr>${i === 0 ? `<td rowspan="${PY[k].length}">${name}</td>` : ""}<td class="num">${r.n}</td><td class="num">${f4(r.p_exceed)}</td><td class="num">${r.mean_sq > 10 ? r.mean_sq.toFixed(0) : f4(r.mean_sq)}</td><td class="num">${f4(r.p_tail)}</td>` +
        (i === 0 ? `<td rowspan="${PY[k].length}">${yes(p === "yes")}</td><td rowspan="${PY[k].length}">${yes(l2 === "yes")}</td><td rowspan="${PY[k].length}">${yes(as === "yes")}</td>` : "") + `</tr>`;
    });
  }
  $("cx-table").innerHTML = t;
})();

// =================================================== section 4: modes demo
for (const [k, S] of Object.entries(SEQS)) $("d-seq").insertAdjacentHTML("beforeend", `<option value="${k}">${S.name}</option>`);
$("d-seq").value = "bern1";
let dSeed = 1;
function drawModes() {
  const key = $("d-seq").value, S = SEQS[key], N = 1000;
  const r = simulateModes(key, 3000, N, dSeed);
  const ns = Array.from({ length: N }, (_, i) => i + 1);
  $("d-note").innerHTML = `ε = ${S.eps}. ${S.note}`;
  let ymax = S.ymax;
  if (!ymax) { ymax = 1; for (const x of r.show) for (const v of x) if (v > ymax) ymax = v; ymax *= 1.05; }
  const ymin = key === "mean" ? -S.ymax : -0.05 * ymax;
  const series = r.show.map((x, i) => ({ type: "line", x: ns, y: Array.from(x), color: i === 0 ? COLORS.orange : COLORS.sky, alpha: i === 0 ? 1 : 0.35, width: i === 0 ? 2 : 1 }));
  series.push(series.shift());
  series.push({ type: "hline", y: S.eps, color: COLORS.white, dash: [5, 4] });
  if (key === "mean") series.push({ type: "hline", y: -S.eps, color: COLORS.white, dash: [5, 4] });
  plot($("d-paths"), { x: [1, N], y: [ymin, ymax], xlog: true, xlabel: "n (log scale)", series });
  plot($("d-crit"), {
    x: [1, N], y: [0, 1.05], xlog: true, xlabel: "n (log scale)",
    series: [
      { type: "line", x: ns, y: Array.from(r.sq).map((v) => (v <= 1.2 ? v : NaN)), color: COLORS.white, width: 1.5 },
      { type: "line", x: ns, y: Array.from(r.exceed), color: COLORS.sky, width: 2 },
      { type: "line", x: ns, y: Array.from(r.tailExceed), color: COLORS.orange, width: 2.5 },
    ],
  });
  let t = `<tr><th class="num">n</th><th class="num">P(|X<sub>n</sub>| &gt; ε)</th><th class="num">E[X<sub>n</sub>²]</th><th class="num">P(sup<sub>n≤k≤1000</sub>|X<sub>k</sub>| &gt; ε)</th></tr>`;
  for (const n of [10, 50, 100, 300]) t += `<tr><td class="num">${n}</td><td class="num">${f4(r.exceed[n - 1])}</td><td class="num">${r.sq[n - 1] > 10 ? r.sq[n - 1].toFixed(1) : f4(r.sq[n - 1])}</td><td class="num">${f4(r.tailExceed[n - 1])}</td></tr>`;
  t += `<tr class="best"><td>converges?</td><td>in P: ${yes(S.modes.p)}</td><td>in L²: ${yes(S.modes.l2)}</td><td>a.s.: ${yes(S.modes.as)}</td></tr>`;
  $("d-table").innerHTML = t;
}
$("d-seq").onchange = drawModes;
$("d-run").onclick = () => { dSeed++; drawModes(); };

// =================================================== section 5: LLN table
$("lln-table").innerHTML = `<tr><th class="num">n</th><th class="num">P(|X̄<sub>n</sub> − ½| &gt; 0.02)</th><th class="num">Chebyshev bound</th><th class="num">E[(X̄<sub>n</sub> − ½)²]</th><th class="num">σ²/n</th><th class="num">P(sup<sub>n≤k≤10⁴</sub>|X̄<sub>k</sub> − ½| &gt; 0.02)</th></tr>` +
  PY.lln.map((r) => `<tr><td class="num">${r.n.toLocaleString("en-US")}</td><td class="num">${f4(r.p_exceed)}</td><td class="num">${f4(r.cheb)}</td><td class="num">${r.mse.toExponential(3)}</td><td class="num">${r.mse_th.toExponential(3)}</td><td class="num">${f4(r.p_tail)}</td></tr>`).join("");

// =================================================== section 6: CLT
const phi = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
function drawCLT() {
  const n = Math.round(10 ** (1 + +$("c-n").value));
  $("c-nv").textContent = n.toLocaleString("en-US");
  const M = 2000, { a, b } = cltPairs(n, M, 7 + n);
  plot($("c-scatter"), { x: [-4, 4], y: [-4, 4], xlabel: "Z_n", series: [
    { type: "fn", f: (x) => x, color: COLORS.white, dash: [5, 4] },
    { type: "points", x: Array.from(a), y: Array.from(b), color: COLORS.sky, r: 1.6, alpha: 0.6 },
  ] });
  const h = histogram(Array.from(a), null, 40);
  plot($("c-hist"), { x: [-4, 4], y: [0, 0.5], series: [{ type: "bars", x: h.x, y: h.y, w: h.w, color: COLORS.sky, alpha: 0.85 }, { type: "fn", f: phi, color: COLORS.orange, width: 2 }] });
  let ma = 0, mb = 0; for (let i = 0; i < M; i++) { ma += a[i]; mb += b[i]; } ma /= M; mb /= M;
  let sab = 0, saa = 0, sbb = 0, far = 0;
  for (let i = 0; i < M; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; if (Math.abs(a[i] - b[i]) > 0.5) far++; }
  $("c-out").innerHTML = `corr(Z_n, Z_2n) = <span class="k">${(sab / Math.sqrt(saa * sbb)).toFixed(4)}</span> (theory 1/√2 = 0.7071) · Var(Z_n) = ${(saa / (M - 1)).toFixed(3)} · P(|Z_n − Z_2n| > 0.5) = ${(far / M).toFixed(3)} (limit 0.514)`;
}
$("c-n").oninput = drawCLT;
$("clt-table").innerHTML = `<tr><th class="num">n</th><th class="num">corr(Z<sub>n</sub>, Z<sub>2n</sub>)</th><th class="num">Var(Z<sub>n</sub>)</th><th class="num">P(|Z<sub>n</sub> − Z<sub>2n</sub>| &gt; 0.5)</th><th class="num">limit</th></tr>` +
  PY.clt.map((r) => `<tr><td class="num">${r.n.toLocaleString("en-US")}</td><td class="num">${r.corr.toFixed(4)}</td><td class="num">${r.var_zn.toFixed(4)}</td><td class="num">${r.p_far.toFixed(4)}</td><td class="num">${r.p_far_th.toFixed(4)}</td></tr>`).join("") +
  `<tr class="best"><td>theory</td><td class="num">0.7071</td><td class="num">1</td><td colspan="2"></td></tr>`;

// =================================================== section 8: n^alpha
const NA = 5000;
const aRuns = (() => {                          // 300 runs of S_n - n/2 for Uniform(0,1), shared by all alphas
  const rng = makeRng(99), runs = [];
  for (let m = 0; m < 300; m++) { const s = new Float64Array(NA); let c = 0; for (let n = 1; n <= NA; n++) { c += rng() - 0.5; s[n - 1] = c; } runs.push(s); }
  return runs;
})();
function drawAlpha() {
  const al = +$("a-alpha").value;
  $("a-av").textContent = al.toFixed(2);
  const ns = Array.from({ length: NA }, (_, i) => i + 1);
  const series = aRuns.slice(0, 20).map((s, i) => ({ type: "line", x: ns, y: Array.from(s, (v, k) => v / Math.pow(k + 1, al)), color: i === 0 ? COLORS.orange : COLORS.sky, alpha: i === 0 ? 1 : 0.4, width: i === 0 ? 2 : 1 }));
  series.push(series.shift());
  const sd = (n) => Math.sqrt(1 / 12) * Math.pow(n, 0.5 - al);
  series.push({ type: "fn", f: (n) => 2 * sd(n), color: COLORS.white, dash: [5, 4] }, { type: "fn", f: (n) => -2 * sd(n), color: COLORS.white, dash: [5, 4] });
  const lim = Math.max(2.5 * sd(1), 2.5 * sd(NA));
  plot($("a-paths"), { x: [1, NA], y: [-lim, lim], xlog: true, xlabel: "n (log scale)", series });
  const pts = [10, 30, 100, 300, 1000, 3000, 5000];
  const v = pts.map((n) => { let s2 = 0; for (const r of aRuns) s2 += (r[n - 1] / Math.pow(n, al)) ** 2; return s2 / aRuns.length; });
  const th = (n) => Math.pow(n, 1 - 2 * al) / 12;
  plot($("a-var"), { x: [5, 1e4], y: [Math.min(th(1e4), th(5)) / 5, Math.max(th(1e4), th(5)) * 5], xlog: true, ylog: true, xlabel: "n",
    series: [{ type: "fn", f: th, color: COLORS.white, dash: [5, 4] }, { type: "points", x: pts, y: v, color: COLORS.orange, r: 5 }] });
  const word = al < 0.45 ? "the variance grows like n^(1−2α): too little normalization, the fluctuations blow up" : al > 0.55 ? (al >= 0.95 ? "this is the sample mean: the variance goes to 0 like 1/n (LLN)" : "the variance goes to 0: the limit is the constant 0") : "α = ½: the variance stays at σ² = 1/12 ≈ 0.0833, the scale of the CLT";
  $("a-out").innerHTML = `Var((S_n − nμ)/n^α) = σ²·n^(1−2α) = n^(${(1 - 2 * al).toFixed(2)})/12 → ${word}`;
}
$("a-alpha").oninput = drawAlpha;
$("sc-table").innerHTML = `<tr><th>α</th><th class="num">n = 10</th><th class="num">100</th><th class="num">1,000</th><th class="num">10,000</th><th>Python, 5,000 runs: Var((S<sub>n</sub> − n/2)/n<sup>α</sup>)</th></tr>` +
  PY.scaling.map((r) => `<tr${r.alpha === 0.5 ? ' class="best"' : ""}><td>${r.alpha}</td>${r.var.map((v) => `<td class="num">${v.toExponential(3)}</td>`).join("")}<td>${r.alpha === 0.5 ? "constant ≈ 1/12" : r.alpha < 0.5 ? "×√10 per decade" : r.alpha === 1 ? "÷10 per decade (σ²/n)" : "÷√10 per decade"}</td></tr>`).join("");

// =================================================== section 9: dependence
$("ar-table").innerHTML = `<tr><th class="num">ρ</th><th class="num">n·Var(X̄<sub>n</sub>), simulated (n = 2,000)</th><th class="num">(1+ρ)/(1−ρ)</th><th class="num">effective sample size</th></tr>` +
  PY.ar1.map((r) => `<tr><td class="num">${r.rho}</td><td class="num">${r.n_var.toFixed(3)}</td><td class="num">${r.theory.toFixed(3)}</td><td class="num">${Math.round(r.ess).toLocaleString("en-US")} of 2,000</td></tr>`).join("");
const iKind = seg($("i-kind"), drawIndep);
function drawIndep() {
  const kind = iKind.get(), rho = +$("i-rho").value;
  $("i-rv").textContent = rho.toFixed(2);
  $("i-rho-wrap").style.opacity = kind === "ar" ? 1 : 0.35;
  const N = 2000, r = runningMeans(kind, rho, N, 600, 11);
  const ns = Array.from({ length: N }, (_, i) => i + 1);
  const series = r.paths.map((p, i) => ({ type: "line", x: ns, y: Array.from(p), color: i === 0 ? COLORS.orange : COLORS.sky, alpha: i === 0 ? 1 : 0.4, width: i === 0 ? 2 : 1 }));
  series.push(series.shift());
  series.push({ type: "fn", f: (n) => 2 / Math.sqrt(n), color: COLORS.white, dash: [5, 4] }, { type: "fn", f: (n) => -2 / Math.sqrt(n), color: COLORS.white, dash: [5, 4] });
  plot($("i-paths"), { x: [1, N], y: [-2.5, 2.5], xlog: true, xlabel: "n (log scale)", series });
  let s2 = 0, out = 0;
  for (const f of r.finals) { s2 += f * f; if (Math.abs(f) > 2 / Math.sqrt(N)) out++; }
  if (kind === "ar") {
    $("i-out").innerHTML = `600 runs: n·Var(X̄_n) at n = 2,000 ≈ <span class="k">${(N * s2 / r.finals.length).toFixed(2)}</span> (theory (1+ρ)/(1−ρ) = ${((1 + rho) / (1 - rho)).toFixed(2)}, i.i.d. = 1) · ` +
      `runs outside the i.i.d. band ±2/√n: ${(100 * out / r.finals.length).toFixed(1)}% (≈ 4.6% if the data were independent)`;
  } else {
    const sorted = Float64Array.from(r.finals).sort();
    $("i-out").innerHTML = `Cauchy: the running means never settle. 600 runs at n = 2,000: interquartile range of X̄_n = <span class="k">${(sorted[449] - sorted[149]).toFixed(2)}</span> (same as one single observation: 2), ` +
      `outside ±2/√n: ${(100 * out / r.finals.length).toFixed(1)}%`;
  }
}
$("i-rho").oninput = drawIndep;

// =================================================== init
drawModes(); drawCLT(); drawAlpha(); drawIndep();
let rt;
window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { drawModes(); drawCLT(); drawAlpha(); drawIndep(); }, 150); });
