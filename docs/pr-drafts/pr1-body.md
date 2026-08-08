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
- pydantic wheel releases: https://github.com/pydantic/pydantic/releases
