# Package Assessment Prompt

Use this prompt to assess the readiness of a Python package for use in the MagnetDB ecosystem.
Run it in each package repository — either by pasting it into a Claude conversation with the
codebase in context, or by using it as a checklist for manual review.

---

## Prompt

Analyse this repository and answer the following questions as concisely as possible.
Return your answer as a **YAML block** with the exact keys shown below.

```yaml
package: <name>
description: <one sentence — what does this package do?>
tests:
  present: <true|false>
  kind: <none | manual scripts | unittest | pytest | pytest+coverage>
  coverage_estimate: <none | low (<30%) | medium (30–70%) | high (>70%) | unknown>
documentation:
  readme: <none | minimal | decent | comprehensive>
  api_docs: <none | docstrings only | sphinx/mkdocs | hosted>
  usage_examples: <none | inline in readme | notebooks | dedicated docs>
ci:
  present: <true|false>
  runs_tests: <true|false>
  builds_package: <true|false>
  platform: <none | GitHub Actions | GitLab CI | other>
installability:
  pip_install: <works | requires manual steps | broken | unknown>
  extra_dependencies: <none | system packages | compiled extensions | complex>
  packaging: <none | setup.py only | pyproject.toml | published to PyPI>
api_stability:
  status: <experimental | unstable | mostly stable | stable>
  breaking_changes_risk: <high | medium | low>
maintenance:
  active: <true|false|unknown>
  last_commit_approx: <e.g. "2 months ago" or a year>
  bus_factor: <solo | small team | team>
overall_readiness:
  score: <1–5>   # 1=prototype, 3=usable with caveats, 5=production-ready
  blocker: <one sentence on the main gap, or "none">
```

---

## Scoring guide for `overall_readiness.score`

| Score | Meaning |
|-------|---------|
| 1 | Prototype — exploratory code, not intended for reuse yet |
| 2 | Early-stage — works for the author, not reliably usable by others |
| 3 | Usable with caveats — installable, functional, but gaps in tests or docs |
| 4 | Good — tested, documented, stable API; minor rough edges |
| 5 | Production-ready — CI green, docs complete, API stable, actively maintained |

---

## Usage

Once you have YAML blocks for all packages in the MagnetDB ecosystem, paste them
together into a single message to generate the maturity heatmap visual.

Packages to assess:

- `python_magnetgeo`
- `python-magnettools`
- `python_magnetrun`
- `python_magnetgmsh`
- `hifimagnet-salome`
- `python_magnetsetup`
- `python_magnetworkflow`
- `hifimagnet-paraview`
- `magnet-scipy`
- `python_magnetcooling`
- `python_magnetapi`
