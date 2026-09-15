"""Regenerates docs/figures/pca_approximation.png with 256x256 images (instead of 64x64)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
FIGS = os.path.join(HERE, "..", "docs", "figures")
sys.path.insert(0, SRC)

import preprocessing_knn as S  # noqa: E402

TS = (256, 256)

print("Loading dataset...")
bw_imgs, bw_dogs, labels, label_names = S.read_and_crop_db(color=False)
assert bw_dogs is not None, "SmallDB not found"

print(f"Resizing {len(bw_dogs)} images to {TS} (continuous padding)...")
resized = S.get_resized_db(bw_dogs, target_size=TS, pad_type="continuous")
data_mtx = S.convert_ndarrays2data_matrix(resized)

data_train, data_test, y_train, y_test = S.train_test_split(
    data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
)

print(f"Fitting PCA on {data_train.shape} ...")
pca = S.my_PCA(data_train)

out = os.path.join(FIGS, "pca_approximation.png")
S.display_pca_approx(data_mtx[0], pca, target_size=TS, fig_path=out)
print("OK ->", out)
