# Kelayakan Beasiswa (Decision Tree)
- data.csv     : dataset (ipk, penghasilan, tanggungan, prestasi, layak)
- train.py     : melatih Decision Tree + evaluasi, menghasilkan model.json
- model.json   : pohon keputusan hasil latih
- index.html   : halaman web (membaca model.json)
Deploy: upload folder ini ke Vercel (framework: Other). Ganti data: edit data.csv lalu `python train.py`.
