"""SVM pipeline: scikit-learn Pipeline and FeatureUnion, grid search, OvO versus OvR.

Requires the arrays cached in cache/ by preprocessing_knn.py.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# 1. Dynamically resolve paths exactly like preprocessing_knn.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "cache")

FIGS = os.path.join(PROJECT_DIR, "docs", "figures")
os.makedirs(FIGS, exist_ok=True)


# Define a local helper to load pickled files safely
def load_dict(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# 2. Reuse the pipeline utilities of the first script
from preprocessing_knn import TARGET_SIZE, compute_hog  # noqa: E402

# Define full absolute paths to cached arrays
X_train_path = os.path.join(DATA_DIR, "X_train_standard.npy")
X_test_path = os.path.join(DATA_DIR, "X_test_standard.npy")
y_train_path = os.path.join(DATA_DIR, "y_train_standard.npy")
y_test_path = os.path.join(DATA_DIR, "y_test_standard.npy")
lbl_names_path = os.path.join(DATA_DIR, "lbl_names.npy")
labels_path = os.path.join(DATA_DIR, "labels.npy")

# 3. Define or reload precalculated vector states from Lab 1 using robust paths
if os.path.exists(X_train_path):
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    label_names = load_dict(lbl_names_path)
    labels = np.load(labels_path)
else:
    raise FileNotFoundError(
        f"Cache arrays not found at: {os.path.abspath(X_train_path)}\n"
        "Please run src/preprocessing_knn.py first to generate them."
    )


# =========================================================================
# TASK 1: QUANTIFYING TRAIN/TEST DISTRIBUTION RESEMBLANCE
# =========================================================================
# Prevent the "Time Machine Effect". We must ensure our random splits
# have similar class distributions before applying any transformations.

### STUDENT IMPLEMENTATION START ###

# TODO: 1. Calculate the empirical distribution arrays (class probabilities) for y_train and y_test.
#          Hint: Use np.unique(..., return_counts=True) and divide by the total length.
# TODO: 2. Implement the Cosine Similarity metric to quantify the resemblance:
#          Formula: (A dot B) / (norm(A) * norm(B))
# What other similarity metrics could you have used here? (e.g., KL Divergence, Wasserstein Distance, etc.)
# How would you implement them mathematically?

_, train_counts = np.unique(y_train, return_counts=True)
_, test_counts = np.unique(y_test, return_counts=True)

P_train = train_counts / len(y_train)
P_test = test_counts / len(y_test)

# Formula: (A dot B) / (norm(A) * norm(B))
dot_product = np.dot(P_train, P_test)
norm_train = np.linalg.norm(P_train)
norm_test = np.linalg.norm(P_test)

similarity_score = dot_product / (norm_train * norm_test)

### STUDENT IMPLEMENTATION END ###
print(f"   - Train/Test Resemblance (Cosine Similarity): {similarity_score:.6f}")


# =========================================================================
# TASK 2: CUSTOM PIPELINE TRANSFORMER WRAPPERS
# =========================================================================
# The "Lego Block" Philosophy: We wrap our custom functions into classes
# that inherit from BaseEstimator and TransformerMixin.


class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, nb_h_cells=4, nb_w_cells=4, nb_bins=8):
        self.nb_h_cells = nb_h_cells
        self.nb_w_cells = nb_w_cells
        self.nb_bins = nb_bins

    def fit(self, X, y=None):
        return self  # HOG requires no training, so fit does nothing

    def transform(self, X):
        return np.array(
            [compute_hog(img.reshape(TARGET_SIZE), self.nb_h_cells, self.nb_w_cells, self.nb_bins) for img in X]
        )


class PCAInfoPreprocessing(BaseEstimator, TransformerMixin):
    """
    In Module 1, class-specific PCA was computed manually via loops.
    To prevent Data Leakage, we automate this inside a scikit-learn transformer.
    """

    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca_per_class = []

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Supervised target array 'y' is required to fit class-specific subspaces.")

        ### STUDENT IMPLEMENTATION START ###
        # TODO: 1. Clear historical instances tracked in self.pca_per_class.
        self.pca_per_class = []
        # TODO: 2. Iterate through unique class identifiers present within vector 'y'.
        classes_uniques = np.unique(y)
        for cl in sorted(classes_uniques):
            # TODO: 3. Isolate matching instance data slices from 'X', train separate PCA models,
            #          and append each fitted model to self.pca_per_class.
            X_classe = X[y == cl]
            pca_modele = PCA(n_components=self.n_components)
            pca_modele.fit(X_classe)
            self.pca_per_class.append(pca_modele)

        ### STUDENT IMPLEMENTATION END ###
        return self

    def transform(self, X):
        out = np.zeros((len(X), 0))

        ### STUDENT IMPLEMENTATION START ###
        # TODO: 1. Iterate through the fitted PCA models saved in self.pca_per_class.
        # TODO: 2. Transform the input matrix 'X' using each PCA model.
        # TODO: 3. Horizontally stack the outputs together using np.hstack.
        projections = []
        for pca_modele in self.pca_per_class:
            X_projette = pca_modele.transform(X)
            projections.append(X_projette)

        out = np.hstack(projections)
        ### STUDENT IMPLEMENTATION END ###
        return out


# =========================================================================
# TASK 3: END-TO-END AUTOMATED PIPELINE ARCHITECTURES (See Slide 5)
# =========================================================================
print("\n Step 2: Assembling automated Pipeline and FeatureUnion components...")

### STUDENT IMPLEMENTATION START ###
# TODO: 1. Construct a 'FeatureUnion' merging 'PCAInfoPreprocessing' and 'EdgeInfoPreprocessing'.
#          Name it 'all_features'.
# TODO: 2. Build the end-to-end processing pipeline using scikit-learn's 'Pipeline' class.
#          The sequence MUST be: MinMaxScaler -> FeatureUnion -> StandardScaler -> SVC(kernel='linear').
#          Name it 'pipeline_svc'.

all_features = FeatureUnion([("pca", PCAInfoPreprocessing(n_components=5)), ("edge", EdgeInfoPreprocessing())])

pipeline_svc = Pipeline(
    [
        ("minmax", MinMaxScaler()),
        ("features", all_features),
        ("scaler", StandardScaler()),
        # class_weight='balanced' : équilibre l'importance des classes selon leur nombre d'échantillons
        # dans le dataset (utile car on a 152 Chihuahuas vs 300 Malamutes, ce qui évite de biaiser le SVM)
        ("classifier", SVC(kernel="linear", class_weight="balanced", random_state=42)),
    ]
)


# =========================================================================
# TASK 4: HYPERPARAMETER OPTIMIZATION SELECTION SPACE (See Slide 6)
# =========================================================================
print("\n Step 3: Tuning via GridSearchCV...")

if pipeline_svc is not None:
    # Switch to RBF kernel for complex boundary mapping
    pipeline_svc.set_params(classifier__kernel="rbf")

    ### STUDENT IMPLEMENTATION START ###
    # Grille de recherche d'hyperparamètres étendue (param_grid)
    # - n_components : nombre de composantes PCA extraites par classe
    # - C : paramètre de régularisation (compromis entre marge et erreurs de classification)
    # - gamma : coefficient du noyau RBF (influence d'un seul échantillon d'apprentissage)
    param_grid = {
        "features__pca__n_components": [3, 5, 10],
        "classifier__C": [1, 5, 10, 20],
        "classifier__gamma": [0.002, 0.005, 0.01, 0.02, 0.1],
    }
    # Fill with proper keys (e.g., 'features__pca__n_components') and value lists

    ### STUDENT IMPLEMENTATION END ###

    # We use 3-Fold Cross Validation to test the parameters safely
    grid_search = GridSearchCV(pipeline_svc, param_grid=param_grid, cv=3, n_jobs=-1)

    print("   Training the Grid Search ")
    # Uncomment the line below once your pipeline and param_grid are built!
    grid_search.fit(X_train, y_train)
    print(f"    Optimal Parameters Identified: {grid_search.best_params_}")
    print(f"    Generalization Score on Testing Partition: {grid_search.score(X_test, y_test) * 100:.2f}%")


# =========================================================================
# TASK 5: MULTICLASS STRATEGY COMPARISON (See Slides 7 & 8)
# =========================================================================
# Support Vector Machines are binary. Let's compare the "Round Robin" (OvO)
# strategy against the "Me Against the World" (OvR) strategy.
print("\n Step 4: Comparing  OvO vs. OvR ")

# TODO: 1. Build a new pipeline named 'pipeline_ovo'. Use the same scaling and feature
#          steps as before, but wrap the final SVC inside a OneVsOneClassifier().
# TODO: 2. Fit 'pipeline_ovo' on the training data and score it on the test data.
# TODO: 3. Repeat the exact same process for a new pipeline named 'pipeline_ovr',
#          but use a OneVsRestClassifier() instead.

# --- OvO Linear Strategy ---
pipeline_ovo_linear = Pipeline(
    [
        ("minmax", MinMaxScaler()),
        ("features", all_features),
        ("scaler", StandardScaler()),
        # class_weight='balanced' : équilibre l'importance des classes selon leur nombre d'échantillons
        # dans le dataset (utile car on a 152 Chihuahuas vs 300 Malamutes, ce qui évite de biaiser le SVM)
        ("classifier", OneVsOneClassifier(SVC(kernel="linear", class_weight="balanced", random_state=42))),
    ]
)
pipeline_ovo_linear.fit(X_train, y_train)
ovo_linear_score = pipeline_ovo_linear.score(X_test, y_test)


# --- OvR Linear Strategy ---
pipeline_ovr_linear = Pipeline(
    [
        ("minmax", MinMaxScaler()),
        ("features", all_features),
        ("scaler", StandardScaler()),
        # class_weight='balanced' : équilibre l'importance des classes selon leur nombre d'échantillons
        # dans le dataset (utile car on a 152 Chihuahuas vs 300 Malamutes, ce qui évite de biaiser le SVM)
        ("classifier", OneVsRestClassifier(SVC(kernel="linear", class_weight="balanced", random_state=42))),
    ]
)
pipeline_ovr_linear.fit(X_train, y_train)
ovr_linear_score = pipeline_ovr_linear.score(X_test, y_test)


# --- OvO RBF Strategy (Tuned) ---
best_C = grid_search.best_params_["classifier__C"]
best_gamma = grid_search.best_params_["classifier__gamma"]
best_pca_n = grid_search.best_params_["features__pca__n_components"]

all_features_rbf = FeatureUnion(
    [
        ("pca", PCAInfoPreprocessing(n_components=best_pca_n)),
        ("edge", EdgeInfoPreprocessing(nb_h_cells=4, nb_w_cells=4, nb_bins=8)),
    ]
)

pipeline_ovo_rbf = Pipeline(
    [
        ("minmax", MinMaxScaler()),
        ("features", all_features_rbf),
        ("scaler", StandardScaler()),
        # class_weight='balanced' : équilibre l'importance des classes selon leur nombre d'échantillons
        # dans le dataset (utile car on a 152 Chihuahuas vs 300 Malamutes, ce qui évite de biaiser le SVM)
        (
            "classifier",
            OneVsOneClassifier(SVC(kernel="rbf", C=best_C, gamma=best_gamma, class_weight="balanced", random_state=42)),
        ),
    ]
)
pipeline_ovo_rbf.fit(X_train, y_train)
ovo_rbf_score = pipeline_ovo_rbf.score(X_test, y_test)


# --- OvR RBF Strategy (Tuned) ---
pipeline_ovr_rbf = Pipeline(
    [
        ("minmax", MinMaxScaler()),
        ("features", all_features_rbf),
        ("scaler", StandardScaler()),
        # class_weight='balanced' : équilibre l'importance des classes selon leur nombre d'échantillons
        # dans le dataset (utile car on a 152 Chihuahuas vs 300 Malamutes, ce qui évite de biaiser le SVM)
        (
            "classifier",
            OneVsRestClassifier(
                SVC(kernel="rbf", C=best_C, gamma=best_gamma, class_weight="balanced", random_state=42)
            ),
        ),
    ]
)
pipeline_ovr_rbf.fit(X_train, y_train)
ovr_rbf_score = pipeline_ovr_rbf.score(X_test, y_test)


print(f"   One-vs-One (OvO) Linear Score: {ovo_linear_score * 100:.2f}%")
print(f"   One-vs-Rest (OvR) Linear Score: {ovr_linear_score * 100:.2f}%")
print(f"   One-vs-One (OvO) RBF Score (C={best_C}, gamma={best_gamma}): {ovo_rbf_score * 100:.2f}%")
print(f"   One-vs-Rest (OvR) RBF Score (C={best_C}, gamma={best_gamma}): {ovr_rbf_score * 100:.2f}%")


# =========================================================================
# TASK 6: CONFUSION MATRIX
# =========================================================================
print("\n Step 5: Calculating and Visualizing the Confusion Matrix...")

best_model = pipeline_ovo_rbf
model_name = "OvO RBF (Tuned)"

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
display_labels = [label_names[i] for i in sorted(label_names.keys())]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
plt.title(f"Confusion Matrix - {model_name}")
plt.tight_layout()

cm_path = os.path.join(FIGS, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"   -> Confusion matrix saved: {cm_path}")
