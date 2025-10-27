# Contributing to Automated-EDA

Thanks for your interest in contributing! We welcome issues, bug reports, feature requests, and pull requests.

Please follow these guidelines to help us process contributions quickly and fairly:

1. Search existing issues and PRs to avoid duplicates.
2. Open an issue to discuss major changes before implementing them.
3. Fork the repository and create a feature branch from `main`:

```bash
git checkout -b feature/your-feature
```

4. Follow the repository code style:
- Format with `black` and sort imports with `isort`.
- Add or update type hints and run `mypy` when appropriate.

5. Write tests for new features or bug fixes. Place them under `tests/`.

6. Commit conventionally with clear messages. Example:

```
git commit -m "feat(eda): add preview sampling strategy"
```

7. Push your branch and open a Pull Request against `main`.

We review PRs as quickly as possible. If your PR is large, consider splitting it into smaller commits.

Thanks — your contributions make the project better!
