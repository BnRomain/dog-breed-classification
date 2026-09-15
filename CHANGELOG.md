# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-09-15

### Changed

- Repository renamed `dog-breed-classification` and reorganized: scripts in `src/` (`preprocessing_knn.py`, `svm_grid_search.py`, `transfer_learning.py`), figure and slides scripts in `scripts/`, report, slides, talk script and figures in `docs/`
- Figures are written to `docs/figures/` and committed, the cached arrays go to `cache/`
- Python sources linted and formatted with Ruff
- README rewritten in English: problem, pipeline, results, getting started, repository structure and quality checks

### Added

- Unit tests of the preprocessing, PCA and HOG functions with pytest and coverage
- CI: Ruff, pytest, the classical pipeline run end to end with the figures as artifact, markdownlint and lychee, dependency review and CodeQL
- Dependabot for the Python dependencies and the GitHub Actions, with auto-merge of patch and minor updates once the required checks pass
- MIT License, contributing guide, code of conduct, security policy, citation metadata, issue forms and pull request template
- Pinned dependencies in `requirements.txt` and `requirements-dev.txt`

### Fixed

- `read_db` ignored its `TO_DB` argument and always read the default dataset folder
- The image files were read in file system order, so the train/test split and the scores differed between Windows and Linux: they are now sorted
- The PCA scatter plot relied on a global `labels` variable: it is now a parameter
- The SVM confusion matrix showed class numbers instead of breed names
- Wrong panel title in the padding comparison figure ("Padding Black" for the propagated blur)
- Typos in file names (`tranfer_learning.py`, `pca_approximatrion.png`, `preprocessing_comparaisonV2.png`)

## [1.0.0] - 2026-06-12

Version submitted at the end of the project week.

### Added

- Classical pipeline: bounding box cropping, aspect-preserving resizing with six padding strategies, PCA per class, hand-written HOG, kNN analysis (scaler, bias-variance, ablation)
- SVM pipeline with scikit-learn `Pipeline` and `FeatureUnion`, grid search of the RBF kernel, one-versus-one and one-versus-rest comparison
- Transfer learning with a frozen VGG16 and a trained head (98 % of test accuracy)
- Report, slides and talk script (French)

[Unreleased]: https://github.com/BnRomain/dog-breed-classification/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/BnRomain/dog-breed-classification/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/BnRomain/dog-breed-classification/releases/tag/v1.0.0
