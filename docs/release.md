# Release guide — for maintainers

The release process below is **not** an automatic publication to GitHub or PyPI.

1. Review `LICENSE` and ownership/credit with the repository maintainers and coauthors.
2. Confirm intended package name availability and package/project ownership before PyPI publication.
3. Review `SECURITY.md`, `CHANGELOG.md`, package metadata and contact channels; add verified project URLs when available.
4. `python -m pip install -e '.[dev,all]'`
5. `python -m pytest -q --cov=stochact --cov-report=term --cov-fail-under=80`
6. `python -m ruff check --select E4,E7,E9 stochact tests examples`
7. `python -m build && python -m twine check dist/*`
8. Install the wheel in a clean virtual environment, from a directory outside the source tree. Run classification, segmentation and ViT example smoke tests.
9. Let the GitHub Actions matrix run on Linux, Windows, and macOS. For CUDA claims, also run the GPU-conditional test on a real GPU runner.
10. Tag only after verifying all results, then explicitly authorize any PyPI publishing action.

Do not invent paper reproduction metrics. The library is designed as a general-purpose research toolkit. The available tests and examples have been validated on CPU. GitHub workflows are provided but have not yet executed on GitHub. Complete CI and relevant GPU validation before advertising broad platform compatibility.
