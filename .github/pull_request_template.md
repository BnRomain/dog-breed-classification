## Summary

<!-- What does this pull request change, and why? Link the related issue, for example "Closes #12". -->

## Type of change

- [ ] Bug fix
- [ ] New feature or new experiment
- [ ] Documentation
- [ ] CI, dependencies or tooling

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `python -m pytest` passes and the coverage stays above the threshold
- [ ] The scripts still run end to end (`python src/preprocessing_knn.py`, then `python src/svm_grid_search.py`)
- [ ] `npx markdownlint-cli2` and `lychee --offline --include-fragments .` pass (if Markdown files changed)
- [ ] `CHANGELOG.md` is updated under "Unreleased"
