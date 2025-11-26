# Plotting tools for GW pe pipelines

This package offers minimal utilities to:
- handle GW posteriors data from `bilby`, `RIFT` and `bajes` 
- produce publication-level plots

## Installation

To install the package:
```bash
pip install .
```

For development, install with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Development

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. Black is automatically run on commit to format code.

To set up pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

To run the hooks manually on all files:
```bash
pre-commit run --all-files
```

### Code Style

This project uses:
- **Black** for code formatting
- **NumPy/Sphinx style** for docstrings

## References
This is largely a wrapper of `corner`, `pesummary` and (sometimes) `bilby`. If you this package, please cite these libraries:

```
@article{corner,
      doi = {10.21105/joss.00024},
      url = {https://doi.org/10.21105/joss.00024},
      year  = {2016},
      month = {jun},
      publisher = {The Open Journal},
      volume = {1},
      number = {2},
      pages = {24},
      author = {Daniel Foreman-Mackey},
      title = {corner.py: Scatterplot matrices in Python},
      journal = {The Journal of Open Source Software}
    }

@article{HOY2021100765,
  title = {PESummary: The code agnostic Parameter Estimation Summary page builder},
  journal = {SoftwareX},
  volume = {15},
  pages = {100765},
  year = {2021},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2021.100765},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711021000856},
  author = {Charlie Hoy and Vivien Raymond},
  keywords = {Parameter Estimation, Python, html, JavaScript, Software},
}

@article{Ashton:2018jfp,
    author = "Ashton, Gregory and others",
    title = "{BILBY: A user-friendly Bayesian inference library for gravitational-wave astronomy}",
    eprint = "1811.02042",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4365/ab06fc",
    journal = "Astrophys. J. Suppl.",
    volume = "241",
    number = "2",
    pages = "27",
    year = "2019"
}
```
