# PR 1 — Quickstart: flag Python 3.13/3.14 wheel wall

## Target
- **Repo:** `datahub-project/datahub`
- **File:** `docs/quickstart.md`
- **Branch name:** `docs/quickstart-python-version-wall`
- **Base:** `master`

## The edit

**Find** (in the Prerequisites section):

```markdown
- Ensure you have **Python 3.10+** installed & configured. (Check using `python3 --version`).
```

**Replace with:**

```markdown
- Ensure you have **Python 3.10–3.12** installed & configured. (Check using `python3 --version`).

  :::note Python 3.13/3.14
  `acryl-datahub` currently fails to install on Python 3.13/3.14 because `pydantic-core` (a transitive dependency) does not yet publish wheels for these versions, and the sdist build fails on most systems. Use Python 3.12 or earlier until upstream ships wheels — track [pydantic/pydantic](https://github.com/pydantic/pydantic/releases) for wheel availability.
  :::
```

## PR title

```
docs(quickstart): flag Python 3.13/3.14 pydantic-core wheel wall
```

## PR body

```markdown
## What

Prerequisites currently say "Python 3.10+", but `pip install acryl-datahub` fails on Python 3.13 and 3.14 because `pydantic-core` doesn't publish wheels for those versions and the sdist build fails on a stock Windows / Homebrew Python. A first-time user hits this in the first five minutes of the quickstart.

## Why

I hit this building [Ogle](https://github.com/BenDuske/ogle) for the Build with DataHub: Agent Hackathon — see the working around in Ogle's [`docs/live-verification.md`](https://github.com/BenDuske/ogle/blob/main/docs/live-verification.md) and [`docs/DEPLOY.md`](https://github.com/BenDuske/ogle/blob/main/docs/DEPLOY.md). Downgrading to Python 3.12 was the fix; the docs pointed the wrong direction.

## The change

One-line update to Prerequisites: change `Python 3.10+` → `Python 3.10–3.12` and add a short admonition explaining the wheel situation. No code touched, no other pages affected.

## How it was verified

- Reproduced on a stock Windows 11 install with Python 3.14: `pip install acryl-datahub` → `Failed building wheel for pydantic-core`.
- Confirmed clean install on Python 3.12.
- Grepped `docs/` for other "3.10+" mentions; only `quickstart.md` needed touching for this scope.

## Related

- Ogle repo: https://github.com/BenDuske/ogle
- pydantic-core wheels tracker: https://github.com/pydantic/pydantic-core
```

## Checklist before pushing

- [ ] Re-fetch `docs/quickstart.md` from master to confirm the exact "Python 3.10+" line still matches (docs move).
- [ ] Run `datahub` docs lint/build if the repo has one (check `docs-website/` for `yarn lint` or similar).
- [ ] Squash to one commit.
