# Writes js/results.js for the web page (run convergence.py and bayes.py first).
import json
conv = json.load(open("results_convergence.json"))
bay = json.load(open("results_bayes.json"))
model = json.load(open("nb_model.json"))
with open("../js/results.js", "w") as f:
    f.write("// part A: numbers produced by code/convergence.py\nconst PY = " + json.dumps(conv) + ";\n")
    f.write("// part B: numbers produced by code/bayes.py\nconst PYB = " + json.dumps(bay) + ";\n")
    f.write("// naive Bayes model trained on 75% of the SMS Spam Collection (words that appear at least 3 times)\n")
    f.write("const NB = " + json.dumps(model, ensure_ascii=False) + ";\n")
