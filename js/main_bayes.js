// ===================================================================
// main_bayes.js - page logic for part B (Bayes); uses $ and seg from main_conv.js
// ===================================================================
const pct1 = (x) => (100 * x).toFixed(x < 0.001 ? 3 : x < 0.1 ? 2 : 1) + "%";
const fmtP = (x) => (x >= 0.01 ? x.toFixed(3) : x.toExponential(1));
const nf = (x) => Math.round(x).toLocaleString("en-US");

// =================================================== B3: IDS
function drawIDS() {
  const prior = 10 ** +$("i-prior").value, sens = +$("i-sens").value, fpr = 10 ** +$("i-fpr").value;
  $("i-pv").textContent = fmtP(prior); $("i-sv").textContent = sens.toFixed(3); $("i-fv").textContent = fmtP(fpr);
  const post = bayesUpdate(prior, sens, fpr), post2 = bayesUpdate(post, sens, fpr);
  // natural frequencies for 100,000 connections
  const N = 100000, att = N * prior, norm = N - att, tp = att * sens, fp = norm * fpr;
  $("i-tree").innerHTML =
    `<div class="fbox">100,000 connections</div>` +
    `<div class="frow"><div class="fbox orange">${nf(att)} attacks</div><div class="fbox sky">${nf(norm)} normal</div></div>` +
    `<div class="frow"><div class="fbox orange">${nf(tp)} alerts <span class="dim">(${nf(att - tp)} missed)</span></div><div class="fbox sky">${nf(fp)} false alarms <span class="dim">(${nf(norm - fp)} quiet)</span></div></div>` +
    `<div class="fbox white">of ${nf(tp + fp)} alerts, ${nf(tp)} are real → ${pct1(post)}</div>`;
  $("i-out").innerHTML =
    `P(attack | alert) = ${sens.toFixed(3)}·${fmtP(prior)} / (${sens.toFixed(3)}·${fmtP(prior)} + ${fmtP(fpr)}·${fmtP(1 - prior)}) = <span class="k">${post.toFixed(4)}</span>\n` +
    `likelihood ratio = ${(sens / fpr).toFixed(1)} · prior odds ${fmtP(prior / (1 - prior))} → posterior odds ${fmtP(post / (1 - post))}\n` +
    `after a second independent alert: <span class="k">${post2.toFixed(4)}</span>`;
}
["i-prior", "i-sens", "i-fpr"].forEach((id) => ($(id).oninput = drawIDS));
$("i-fpr-table").innerHTML = `<tr><th class="num">false positive rate</th><th class="num">P(attack | alert)</th></tr>` +
  PYB.ids.fpr_table.map((r) => `<tr><td class="num">${r.fpr}</td><td class="num">${r.post.toFixed(4)}</td></tr>`).join("");
$("i-prior-table").innerHTML = `<tr><th class="num">prior P(attack)</th><th class="num">P(attack | alert)</th></tr>` +
  PYB.ids.prior_table.map((r) => `<tr><td class="num">${r.prior}</td><td class="num">${r.post.toFixed(4)}</td></tr>`).join("");

// =================================================== B4: Beta-Binomial
$("b-table").innerHTML = `<tr><th>prior</th><th class="num">prior mean</th><th class="num">weight of prior</th><th>posterior</th><th class="num">posterior mean</th><th class="num">95% credible interval</th><th class="num">P(p &gt; 0.10 | data)</th></tr>` +
  PYB.bb.rows.map((r) => `<tr><td>${r.name}</td><td class="num">${r.prior_mean.toFixed(3)}</td><td class="num">${r.prior_weight.toFixed(2)}</td><td>Beta(${r.post_a}, ${r.post_b})</td><td class="num">${r.post_mean.toFixed(4)}</td><td class="num">[${r.lo.toFixed(3)}, ${r.hi.toFixed(3)}]</td><td class="num">${r.p_above_10.toFixed(3)}</td></tr>`).join("") +
  `<tr class="best"><td>frequentist (no prior)</td><td></td><td></td><td>MLE k/n</td><td class="num">${PYB.bb.mle.toFixed(4)}</td><td class="num">[${PYB.bb.wilson[0].toFixed(3)}, ${PYB.bb.wilson[1].toFixed(3)}] Wilson</td><td></td></tr>`;

function drawBeta() {
  const a = +$("b-a").value, b = +$("b-b").value, n = +$("b-n").value;
  $("b-k").max = n;
  if (+$("b-k").value > n) $("b-k").value = n;
  const k = +$("b-k").value;
  $("b-av").textContent = a; $("b-bv").textContent = b; $("b-nv").textContent = n; $("b-kv").textContent = k;
  const A = a + k, B = b + n - k;
  const lo = betaPpf(0.025, A, B), hi = betaPpf(0.975, A, B);
  const xs = [], pr = [], li = [], po = [];
  for (let i = 1; i < 400; i++) { const p = i / 400; xs.push(p); pr.push(betaPdf(p, a, b)); li.push(n > 0 ? betaPdf(p, k + 1, n - k + 1) : 1); po.push(betaPdf(p, A, B)); }
  const top = Math.min(Math.max(...po, ...li.slice(2, -2), ...pr.slice(2, -2)) * 1.1, 60);
  const shade = [];
  for (let i = 0; i < xs.length; i++) if (xs[i] >= lo && xs[i] <= hi) shade.push([xs[i] - 0.00125, 0, xs[i] + 0.00125, po[i]]);
  plot($("b-chart"), {
    x: [0, 1], y: [0, top], xlabel: "click rate p",
    series: [
      { type: "rects", rects: shade, color: COLORS.orange, alpha: 0.25 },
      { type: "line", x: xs, y: pr, color: COLORS.sky, width: 2 },
      { type: "line", x: xs, y: li, color: COLORS.white, width: 1.5, dash: [5, 4] },
      { type: "line", x: xs, y: po, color: COLORS.orange, width: 2.5 },
    ],
  });
  const w = (a + b) / (a + b + n);
  $("b-out").innerHTML =
    `posterior Beta(${A}, ${B}): mean <span class="k">${(A / (A + B)).toFixed(4)}</span> = ${w.toFixed(2)}·(prior mean ${(a / (a + b)).toFixed(3)}) + ${(1 - w).toFixed(2)}·(MLE ${n ? (k / n).toFixed(3) : "-"})\n` +
    `95% credible interval [${lo.toFixed(3)}, ${hi.toFixed(3)}] · P(p > 0.10 | data) = <span class="k">${(1 - betaCdf(0.1, A, B)).toFixed(3)}</span>`;
}
["b-a", "b-b", "b-n", "b-k"].forEach((id) => ($(id).oninput = drawBeta));
document.querySelectorAll("#beta button[data-ab]").forEach((btn) => (btn.onclick = () => {
  const [a, b] = btn.dataset.ab.split(",").map(Number);
  $("b-a").value = a; $("b-b").value = b; drawBeta();
}));

const PRIOR_NAMES = PYB.bb.rows.map((r) => r.name);
const WCOL = [COLORS.white, COLORS.sky, "#6ee7a8", COLORS.orange];
$("b-wash").innerHTML = `<tr><th class="num">n</th><th class="num">k</th><th class="num">k/n</th>${PRIOR_NAMES.map((n) => `<th class="num">${n.replace(/ Beta/, "<br>Beta")}</th>`).join("")}</tr>` +
  PYB.wash.map((r) => `<tr><td class="num">${nf(r.n)}</td><td class="num">${r.k}</td><td class="num">${r.n ? (r.k / r.n).toFixed(4) : "-"}</td>${PRIOR_NAMES.map((n) => `<td class="num">${r[n].toFixed(4)}</td>`).join("")}</tr>`).join("");
function drawWash() {
  const rows = PYB.wash.filter((r) => r.n > 0);
  const series = [{ type: "hline", y: 0.15, color: COLORS.grey, dash: [3, 3] }];
  PRIOR_NAMES.forEach((n, i) => {
    series.push({ type: "line", x: rows.map((r) => r.n), y: rows.map((r) => r[n]), color: WCOL[i], width: 2 });
    series.push({ type: "points", x: rows.map((r) => r.n), y: rows.map((r) => r[n]), color: WCOL[i], r: 3.5 });
  });
  plot($("b-wash-chart"), { x: [8, 6000], y: [0, 0.42], xlog: true, xlabel: "n (emails)", ylabel: "posterior mean", series });
  $("b-wash-legend").innerHTML = PRIOR_NAMES.map((n, i) => `<span><i style="background:${WCOL[i]}"></i>${n}</span>`).join("") + `<span>dashed: true value 0.15</span>`;
}

// =================================================== B5: spam filter
const MSGS = [
  "URGENT! You have won a £1000 prize. Call now to claim, T&C apply",
  "Hey, are you still coming to dinner later? I'll bring the wine",
  "Text me your number to win",
  "Your account has been suspended. Reply with your PIN to verify your details now",
];
document.querySelectorAll("#spam button[data-msg]").forEach((btn) => (btn.onclick = () => { $("s-text").value = MSGS[+btn.dataset.msg]; drawSpam(); }));
function drawSpam() {
  const prior = 10 ** +$("s-prior").value;
  $("s-pv").textContent = prior.toFixed(prior < 0.01 ? 4 : 3);
  const r = classify($("s-text").value, prior);
  const pTxt = r.p > 0.99999 ? "> 0.99999" : r.p < 0.00001 ? "< 0.00001" : r.p.toFixed(5);
  $("s-out").innerHTML =
    `log prior odds ${r.prior.toFixed(2)} + Σ word log-likelihood ratios ${(r.logOdds - r.prior).toFixed(2)} = log posterior odds <span class="k">${r.logOdds.toFixed(2)}</span>\n` +
    `P(spam | message) = <span class="k">${pTxt}</span> → ${r.p > 0.5 ? '<span class="bad">SPAM</span>' : '<span class="ok">ham</span>'}` +
    `   (${r.terms.filter((t) => !t.known).length} words not in the model, ignored)`;
  const known = r.terms.filter((t) => t.known);
  const mx = Math.max(1, ...known.map((t) => Math.abs(t.llr)));
  $("s-words").innerHTML = r.terms.map((t) => {
    if (!t.known) return `<div class="wb"><span class="ww dim">${t.w}</span><span class="wbar"></span><span class="wv dim">unknown</span></div>`;
    const width = (50 * Math.abs(t.llr)) / mx;
    const style = t.llr > 0 ? `left:50%;width:${width}%;background:var(--orange)` : `right:50%;width:${width}%;background:var(--sky)`;
    return `<div class="wb"><span class="ww">${t.w}</span><span class="wbar"><i style="${style}"></i></span><span class="wv">${t.llr > 0 ? "+" : ""}${t.llr.toFixed(2)} <span class="dim">(spam ${t.spam}, ham ${t.ham})</span></span></div>`;
  }).join("");
}
$("s-text").oninput = drawSpam;
$("s-prior").oninput = drawSpam;
(function spamTables() {
  const E = PYB.nb.eval;
  const rows = [["learned prior", `learned, ${PYB.nb.prior.toFixed(3)}`], ["prior 0.5", "0.5"], ["prior 0.9", "0.9"], ["prior 0.01", "0.01"], ["prior 0.001", "0.001"], ["no smoothing", "learned, almost no smoothing"]];
  $("s-table").innerHTML = `<tr><th>prior P(spam)</th><th class="num">accuracy</th><th class="num">spam caught</th><th class="num">normal flagged as spam</th><th class="num">precision</th><th class="num">recall</th></tr>` +
    rows.map(([k, lab]) => { const e = E[k]; return `<tr${k === "learned prior" ? ' class="best"' : ""}><td>${lab}</td><td class="num">${(100 * e.acc).toFixed(1)}%</td><td class="num">${e.tp} / ${e.tp + e.fn}</td><td class="num">${e.fp} / ${e.fp + e.tn}</td><td class="num">${e.prec.toFixed(3)}</td><td class="num">${e.rec.toFixed(3)}</td></tr>`; }).join("");
  const wt = (list) => `<tr><th>word</th><th class="num">in spam</th><th class="num">in ham</th><th class="num">log LR</th></tr>` + list.map((r) => `<tr><td>${r.w}</td><td class="num">${r.s}</td><td class="num">${r.h}</td><td class="num">${r.lr > 0 ? "+" : ""}${r.lr.toFixed(2)}</td></tr>`).join("");
  $("s-spammy").innerHTML = wt(PYB.nb.spammy);
  $("s-hammy").innerHTML = wt(PYB.nb.hammy);
})();

// =================================================== B6: normal-normal
function drawNormal() {
  const n = +$("n-n").value, tau = +$("n-tau").value, xb = +$("n-x").value, mu0 = 200, sigma = 60;
  $("n-nv").textContent = n; $("n-tv").textContent = tau; $("n-xv").textContent = xb;
  const prec = 1 / tau ** 2 + n / sigma ** 2, m = (mu0 / tau ** 2 + (n * xb) / sigma ** 2) / prec, s = Math.sqrt(1 / prec);
  const npdf = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
  const se = n > 0 ? sigma / Math.sqrt(n) : Infinity;
  const top = 1.1 * Math.max(npdf(mu0, mu0, tau), npdf(m, m, s), n > 0 ? npdf(xb, xb, se) : 0);
  const series = [{ type: "fn", f: (x) => npdf(x, mu0, tau), color: COLORS.sky, width: 2 }];
  if (n > 0) series.push({ type: "fn", f: (x) => npdf(x, xb, se), color: COLORS.white, width: 1.5, dash: [5, 4] });
  series.push({ type: "fn", f: (x) => npdf(x, m, s), color: COLORS.orange, width: 2.5 });
  plot($("n-chart"), { x: [100, 320], y: [0, top], xlabel: "mean response time μ (ms)", series });
  const w = 1 / tau ** 2 / prec;
  $("n-out").innerHTML = `posterior N(<span class="k">${m.toFixed(2)}</span>, ${s.toFixed(2)}²) · weight of the prior ${w.toFixed(3)}, of the data ${(1 - w).toFixed(3)} · ` +
    `data standard error σ/√n = ${n > 0 ? se.toFixed(2) : "∞"} ms · 95% credible interval [${(m - 1.96 * s).toFixed(1)}, ${(m + 1.96 * s).toFixed(1)}]`;
}
["n-n", "n-tau", "n-x"].forEach((id) => ($(id).oninput = drawNormal));

// =================================================== init
drawIDS(); drawBeta(); drawWash(); drawSpam(); drawNormal();
window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { drawModes(); drawCLT(); drawAlpha(); drawIndep(); drawBeta(); drawWash(); drawNormal(); }, 150); });

const links = [...document.querySelectorAll("nav.toc a")];
const obs = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id)); });
}, { rootMargin: "-40% 0px -55% 0px" });
document.querySelectorAll("main section").forEach((s) => obs.observe(s));
