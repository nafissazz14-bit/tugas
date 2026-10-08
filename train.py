"""Latih Decision Tree untuk kelayakan beasiswa lalu ekspor ke model.json.
Jalankan: python train.py   (butuh pandas & scikit-learn)
Untuk data asli: ganti data.csv dengan kolom yang sama."""
import json, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FITUR = ["ipk", "penghasilan", "tanggungan", "prestasi"]

# 1. Buat data simulasi bila data.csv belum ada
import os
if not os.path.exists("data.csv"):
    rng = np.random.default_rng(42); n = 400
    df = pd.DataFrame({
        "ipk": np.round(rng.uniform(2.0, 4.0, n), 2),
        "penghasilan": np.round(rng.gamma(2.5, 2.2, n), 1),   # juta/bulan
        "tanggungan": rng.integers(0, 6, n),
        "prestasi": rng.choice([0,1,2,3], n, p=[.4,.3,.2,.1]),
    })
    skor = ((df.ipk-2)/2*4 + np.clip((8-df.penghasilan)/8,0,1)*3
            + df.tanggungan*.25 + df.prestasi*.8 + rng.normal(0,.6,n))
    df["layak"] = (skor >= 4.6).astype(int)
    df.to_csv("data.csv", index=False)

df = pd.read_csv("data.csv")
X, y = df[FITUR], df["layak"]

# 2. Bagi data latih 80% / uji 20%
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1, stratify=y)

# 3. Latih Decision Tree (entropy = information gain, kedalaman dibatasi agar tidak overfit)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=8, random_state=1)
clf.fit(Xtr, ytr)

# 4. Evaluasi
pred = clf.predict(Xte)
acc = accuracy_score(yte, pred)
print("Akurasi data uji:", round(acc*100,1), "%")
print(confusion_matrix(yte, pred)); print(classification_report(yte, pred, target_names=["tidak layak","layak"]))

# 5. Ekspor pohon ke JSON untuk dipakai web
t = clf.tree_
def node(i):
    v = t.value[i][0]; 
    if t.children_left[i] == -1:
        return {"leaf": True, "kelas": int(v.argmax()), "n": int(t.n_node_samples[i]),
                "layak": int(round(v[1]*t.n_node_samples[i]/v.sum()))}
    return {"fitur": FITUR[t.feature[i]], "batas": round(float(t.threshold[i]),3),
            "n": int(t.n_node_samples[i]), "kiri": node(t.children_left[i]), "kanan": node(t.children_right[i])}
json.dump({"pohon": node(0), "akurasi": round(acc,4), "n_latih": len(Xtr), "n_uji": len(Xte),
           "pentingnya": dict(zip(FITUR, map(lambda a: round(float(a),3), clf.feature_importances_)))},
          open("model.json","w"), indent=1)
print("model.json tersimpan")
