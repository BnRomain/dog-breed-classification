"""Draws the class balance bar plot (docs/figures/class_balance.png) from the SmallDB counts."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
FIGS = os.path.join(HERE, "..", "docs", "figures")
sys.path.insert(0, SRC)

import preprocessing_knn as S  # noqa: E402

# Counts of SmallDB/Images, in the sorted order of the folders
breeds = [
    ("Chihuahua", 152),
    ("basset", 175),
    ("Kerry_blue_terrier", 179),
    ("groenendael", 150),
    ("malinois", 150),
    ("chow", 196),
]
label_names = {i: name for i, (name, _) in enumerate(breeds)}
labels = []
for i, (_, n) in enumerate(breeds):
    labels += [i] * n

out = os.path.join(FIGS, "class_balance.png")
S.plot_barplot(labels, label_names, fig_path=out)
print("OK ->", out)
