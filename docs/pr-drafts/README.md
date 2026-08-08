# DataHub OSS Contribution PR Drafts

Two docs PRs against [`datahub-project/datahub`](https://github.com/datahub-project/datahub) drafted for the Build with DataHub Agent Hackathon. Both surface real friction hit while building Ogle.

- **PR 1 — `pr1-quickstart-python-version.md`** — quickstart-python-version wall fix (safe layup, ~30-min PR)
- **PR 2 — `pr2-ml-tutorial-read-back.md`** — add a "Read ML Lineage Back" section to the AI/ML tutorial (bigger flex; positions Ogle as the reference implementation)

## How to open them (do this Sat/Sun)

```bash
# One-time: fork datahub-project/datahub to BenDuske on github.com

# Clone the fork
cd ~/dev  # or wherever
git clone https://github.com/BenDuske/datahub.git
cd datahub
git remote add upstream https://github.com/datahub-project/datahub.git
git fetch upstream

# Branch + apply PR 1
git checkout -b docs/quickstart-python-version-wall upstream/master
# apply the edit from pr1-quickstart-python-version.md (Edit tool or hand-edit docs/quickstart.md)
git add docs/quickstart.md
git commit -m "docs(quickstart): flag pydantic-core wheel wall on Python 3.13/3.14"
git push -u origin docs/quickstart-python-version-wall
gh pr create --repo datahub-project/datahub --base master --title "..." --body-file ../ogle/docs/pr-drafts/pr1-body.md

# Same for PR 2 on branch docs/ml-tutorial-read-lineage-back
```

## Verification before submitting

Both PRs cite Ogle. Before opening, sanity-check:

- The exact source line still exists in `docs/quickstart.md` (Prerequisites → "Python 3.10+"). Re-fetch `https://raw.githubusercontent.com/datahub-project/datahub/master/docs/quickstart.md` if it's been a few days.
- The AI/ML tutorial still doesn't have a "Read Back" section — grep for `Read.*Lineage.*Back` in the raw source.
- Ogle's `walker.DataHubBackend.get_dataset_profile` still exists at `src/ogle/walker.py:596-605` (the snippet PR 2 references).

## Devpost submission link

In Ogle's Devpost submission's "OSS contribution" section, paste both PR URLs after opening.
