"""
Homework 7 (part B) - Conditional probability, Bayes' theorem and Bayesian inference.
Python 3 + scipy (only to double check the Beta quantiles).  Run:  python3 bayes.py
"""
import json, math, random, re
from collections import Counter

out = {}

# ------------------------------------------------------------------ 1. IDS alerts and the base rate
def ids_posterior(prior, sens, fpr):
    """P(attack | alert) = P(alert | attack) P(attack) / P(alert)"""
    evidence = sens * prior + fpr * (1 - prior)
    return sens * prior / evidence

print("=== 1. IDS: P(attack | alert)")
base = dict(prior=0.001, sens=0.99, fpr=0.01)
p1 = ids_posterior(**base)
p2 = ids_posterior(p1, base["sens"], base["fpr"])          # second, conditionally independent alert
p3 = ids_posterior(p2, base["sens"], base["fpr"])
print(f"prior 0.1%, sensitivity 99%, false positive rate 1% -> after 1 alert {p1:.4f}, after 2 alerts {p2:.4f}, after 3 {p3:.4f}")
fpr_table = []
for fpr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    fpr_table.append(dict(fpr=fpr, post=ids_posterior(0.001, 0.99, fpr)))
    print(f"  fpr={fpr:<8} P(attack|alert) = {fpr_table[-1]['post']:.4f}")
prior_table = []
for prior in [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    prior_table.append(dict(prior=prior, post=ids_posterior(prior, 0.99, 0.01)))
out["ids"] = dict(p1=p1, p2=p2, p3=p3, fpr_table=fpr_table, prior_table=prior_table)

# ------------------------------------------------------------------ 2. Beta-Binomial: click rate of a phishing simulation
# regularized incomplete beta I_x(a, b) (continued fraction, Numerical Recipes style) and its inverse by bisection
def betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny); h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        delta = d * c; h *= delta
        if abs(delta - 1) < 1e-15: break
    return h

def beta_cdf(x, a, b):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2): return math.exp(lbt) * betacf(a, b, x) / a
    return 1 - math.exp(lbt) * betacf(b, a, 1 - x) / b

def beta_ppf(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if beta_cdf(mid, a, b) < q: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def wilson(k, n, z=1.959963984540054):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

print("\n=== 2. Beta-Binomial: click rate p of a phishing test, k = 9 clicks out of n = 60 emails")
K, N = 9, 60
PRIORS = {
    "uniform Beta(1,1)": (1, 1),
    "weak, from past campaigns Beta(2,18)": (2, 18),
    "strong, from past campaigns Beta(20,180)": (20, 180),
    "wrong and strong Beta(40,60)": (40, 60),
}
bb = []
for name, (a, b) in PRIORS.items():
    A, B = a + K, b + N - K
    r = dict(name=name, a=a, b=b, prior_mean=a / (a + b), post_a=A, post_b=B, post_mean=A / (A + B),
             lo=beta_ppf(0.025, A, B), hi=beta_ppf(0.975, A, B), p_above_10=1 - beta_cdf(0.10, A, B),
             prior_weight=(a + b) / (a + b + N))
    bb.append(r)
    print(f"{name:42s} prior mean {r['prior_mean']:.3f} -> posterior Beta({A},{B}) mean {r['post_mean']:.4f}  95% CrI [{r['lo']:.4f}, {r['hi']:.4f}]  P(p>0.10|data)={r['p_above_10']:.3f}  weight of prior {r['prior_weight']:.2f}")
wl = wilson(K, N)
out["bb"] = dict(k=K, n=N, rows=bb, mle=K / N, wilson=wl)
print(f"frequentist: p_hat = {K / N:.4f}, Wilson 95% CI [{wl[0]:.4f}, {wl[1]:.4f}]")

# prior washing out: data from a true p = 0.15, posterior mean under each prior as n grows
rnd = random.Random(2026)
stream = [1 if rnd.random() < 0.15 else 0 for _ in range(5000)]
wash = []
for n in [0, 10, 30, 100, 300, 1000, 3000, 5000]:
    k = sum(stream[:n])
    row = dict(n=n, k=k)
    for name, (a, b) in PRIORS.items():
        A, B = a + k, b + n - k
        row[name] = A / (A + B)
        row[name + " sd"] = math.sqrt(A * B / ((A + B) ** 2 * (A + B + 1)))
    wash.append(row)
    print(f"n={n:5d} k={k:4d}  " + "  ".join(f"{row[nm]:.4f}" for nm in PRIORS))
out["wash"] = wash

try:
    from scipy.stats import beta as sbeta
    for r in bb:
        print(f"scipy check: [{sbeta.ppf(0.025, r['post_a'], r['post_b']):.6f}, {sbeta.ppf(0.975, r['post_a'], r['post_b']):.6f}] vs mine [{r['lo']:.6f}, {r['hi']:.6f}]")
except ImportError:
    pass

# ------------------------------------------------------------------ 3. naive Bayes spam filter on the SMS Spam Collection
print("\n=== 3. Naive Bayes on the SMS Spam Collection")
rows = [l.rstrip("\n").split("\t", 1) for l in open("../data/sms.tsv", encoding="utf-8")]
rows = [(lab, txt) for lab, txt in rows if lab in ("ham", "spam")]
tok = lambda s: re.findall(r"[a-z0-9£$']+", s.lower())
rnd = random.Random(7); idx = list(range(len(rows))); rnd.shuffle(idx)
cut = int(0.75 * len(rows)); train = [rows[i] for i in idx[:cut]]; test = [rows[i] for i in idx[cut:]]

def fit(data):
    """multinomial naive Bayes: word counts per class, number of messages per class"""
    cnt = {"ham": Counter(), "spam": Counter()}; docs = Counter()
    for lab, txt in data:
        docs[lab] += 1
        cnt[lab].update(tok(txt))
    return cnt, docs

def log_odds(txt, cnt, docs, prior_spam=None, alpha=1.0):
    """log P(spam | message) - log P(ham | message)
       = log prior odds + sum over the words of log P(word | spam) / P(word | ham)   (naive: words independent given the class)"""
    ps = prior_spam if prior_spam is not None else docs["spam"] / (docs["spam"] + docs["ham"])
    V = len(set(cnt["spam"]) | set(cnt["ham"]))
    ts, th = sum(cnt["spam"].values()), sum(cnt["ham"].values())
    lo = math.log(ps) - math.log(1 - ps)
    for w in tok(txt):
        if cnt["spam"][w] + cnt["ham"][w] == 0: continue        # word never seen in training: skip it
        ls = (cnt["spam"][w] + alpha) / (ts + alpha * V)        # Laplace smoothing
        lh = (cnt["ham"][w] + alpha) / (th + alpha * V)
        lo += math.log(ls) - math.log(lh)
    return lo

cnt, docs = fit(train)
prior_spam = docs["spam"] / (docs["spam"] + docs["ham"])
print(f"train: {docs['ham']} ham, {docs['spam']} spam -> prior P(spam) = {prior_spam:.4f};  test: {len(test)} messages")
def evaluate(prior=None, alpha=1.0, thr=0.0):
    tp = fp = tn = fn = 0
    for lab, txt in test:
        pred = log_odds(txt, cnt, docs, prior, alpha) > thr
        if lab == "spam": tp += pred; fn += not pred
        else: fp += pred; tn += not pred
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, acc=(tp + tn) / len(test), prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1))
ev = {}
for name, pr in [("learned prior", None), ("prior 0.5", 0.5), ("prior 0.01", 0.01), ("prior 0.001", 0.001), ("prior 0.9", 0.9)]:
    ev[name] = evaluate(pr)
    e = ev[name]
    print(f"{name:14s} acc {e['acc']:.4f}  spam caught {e['tp']}/{e['tp'] + e['fn']}  ham flagged as spam {e['fp']}/{e['fp'] + e['tn']}  precision {e['prec']:.3f} recall {e['rec']:.3f}")
e0 = evaluate(alpha=1e-9)
print(f"almost no smoothing (alpha=1e-9): acc {e0['acc']:.4f}  spam caught {e0['tp']}  ham flagged {e0['fp']}")
ev["no smoothing"] = e0
# most indicative words (both classes, at least 20 documents)
ratio = []
for w in set(cnt["spam"]) | set(cnt["ham"]):
    tot = cnt["spam"][w] + cnt["ham"][w]
    if tot < 20: continue
    V = len(set(cnt["spam"]) | set(cnt["ham"])); ts, th = sum(cnt["spam"].values()), sum(cnt["ham"].values())
    ls = (cnt["spam"][w] + 1) / (ts + V); lh = (cnt["ham"][w] + 1) / (th + V)
    ratio.append((math.log(ls / lh), w, cnt["spam"][w], cnt["ham"][w]))
ratio.sort()
print("most 'spammy':", [(w, s, h, round(r, 2)) for r, w, s, h in ratio[-12:][::-1]])
print("most 'hammy': ", [(w, s, h, round(r, 2)) for r, w, s, h in ratio[:12]])
out["nb"] = dict(n_train=len(train), n_test=len(test), ham=docs["ham"], spam=docs["spam"], prior=prior_spam, eval=ev,
                 spammy=[dict(w=w, s=s, h=h, lr=r) for r, w, s, h in ratio[-12:][::-1]], hammy=[dict(w=w, s=s, h=h, lr=r) for r, w, s, h in ratio[:12]])
# vocabulary for the browser classifier: words that appear at least 3 times in the training messages
vocab = {w: [cnt["spam"][w], cnt["ham"][w]] for w in set(cnt["spam"]) | set(cnt["ham"]) if cnt["spam"][w] + cnt["ham"][w] >= 3}
out["nb_vocab_size"] = len(vocab)
print("vocabulary exported:", len(vocab), "words")

# ------------------------------------------------------------------ 4. Normal-Normal: average response time of a server (known sigma)
print("\n=== 4. Normal prior, normal likelihood")
mu0, tau0 = 200.0, 20.0           # prior on the mean response time (ms): N(200, 20^2)
sigma = 60.0                      # known spread of single requests (ms)
xbar, n = 236.0, 25
prec = 1 / tau0 ** 2 + n / sigma ** 2
mu_post = (mu0 / tau0 ** 2 + n * xbar / sigma ** 2) / prec
sd_post = math.sqrt(1 / prec)
w_prior = (1 / tau0 ** 2) / prec
print(f"prior N({mu0}, {tau0}^2), n={n}, xbar={xbar}, sigma={sigma} -> posterior N({mu_post:.2f}, {sd_post:.2f}^2), weight of the prior {w_prior:.3f}")
out["nn"] = dict(mu0=mu0, tau0=tau0, sigma=sigma, xbar=xbar, n=n, mu_post=mu_post, sd_post=sd_post, w_prior=w_prior)

json.dump(out, open("results_bayes.json", "w"), indent=1)
V = len(set(cnt["spam"]) | set(cnt["ham"]))
json.dump(dict(spam_docs=docs["spam"], ham_docs=docs["ham"], V=V, spam_words=sum(cnt["spam"].values()), ham_words=sum(cnt["ham"].values()), vocab=vocab), open("nb_model.json", "w"))
