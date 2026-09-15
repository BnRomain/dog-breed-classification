# Contributing

Thank you for your interest in this project. It started as a one-week student
project at Polytech Nice Sophia, and bug reports, new experiments and pull
requests are welcome.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** or **suggest an improvement** with the
  [issue forms](https://github.com/BnRomain/dog-breed-classification/issues/new/choose).
- **Report a security vulnerability** privately, as described in the
  [security policy](SECURITY.md). Please do not open a public issue for it.
- **Open a pull request** for a fix, a test, a new experiment or documentation.

For a larger change, for example a new feature extractor or another model,
please open an issue first so that we can agree on the approach.

## Development setup

Requires Python 3.12.

```bash
git clone https://github.com/BnRomain/dog-breed-classification.git
cd dog-breed-classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m pytest -v                # tests with a coverage report
ruff check .                       # lint
ruff format .                      # format
python src/preprocessing_knn.py    # classical pipeline, first part (about 30 s)
python src/svm_grid_search.py      # second part (about 2 min)
```

The transfer learning script needs `pip install tensorflow` and downloads the
VGG16 weights on first run.

To check the Markdown files like the CI does:

```bash
npx markdownlint-cli2
lychee --offline --include-fragments .
```

## Coding guidelines

The repository provides an [`.editorconfig`](.editorconfig) file: most editors
apply its indentation and whitespace settings automatically.

- The code must pass `ruff check` and be formatted with `ruff format` (a
  Black-compatible style), both configured in [`pyproject.toml`](pyproject.toml).
- `src/preprocessing_knn.py` is both a script and the module shared by the other
  scripts: keep its functions importable (no work at import time outside the
  `if __name__ == "__main__":` block) and add or update a pytest test in
  `tests/` for every change of behavior. The CI fails if the coverage of this
  module drops below 60 %.
- `src/svm_grid_search.py` and `src/transfer_learning.py` run the whole
  pipeline at import time: they are exercised by the `pipeline` job of the CI,
  not by pytest.
- Figures go to `docs/figures/` and are committed, so that the report, the
  slides and the README always match the code. Rerun the scripts after a change
  that affects them, and update the numbers of the README and the report.
- Comments explain the reasoning, not what the code already says.
- Pin new dependencies to an exact version in `requirements.txt` (or
  `requirements-dev.txt` for development tools) so that Dependabot can track them.

## Pull request process

1. Create a branch from `main` with a descriptive name, for example
   `fix/hog-bin-overflow` or `docs/results-table`.
2. Keep commits focused, with a short summary in the imperative mood
   (for example "Add a test for the reflect padding").
3. Open a pull request against `main`, fill in the template and add a label
   (`bug`, `enhancement`, `documentation`...): labels sort the release notes.
4. The `main` branch is protected: a pull request can only be merged once the
   required checks (`python`, `pipeline`, `docs` and `dependency-review`) pass
   and the branch is up to date with `main`. CodeQL also analyzes every pull
   request.
5. Update the documentation (README, [wiki](https://github.com/BnRomain/dog-breed-classification/wiki))
   when the usage or the results change, and `CHANGELOG.md` under "Unreleased".

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (`2.0.0`): incompatible change, for example to the layout of the
  dataset, the cached arrays or the signature of the shared functions;
- **MINOR** (`1.1.0`): new backward-compatible feature, such as a new padding
  strategy, a new experiment or a new figure;
- **PATCH** (`1.0.1`): backward-compatible bug fix.

Releases are published from `main` with a `vX.Y.Z` tag. GitHub generates their
notes from the merged pull requests, grouped by label as configured in
[`.github/release.yml`](.github/release.yml), and the `version` field of
[`CITATION.cff`](CITATION.cff) is updated at the same time.
