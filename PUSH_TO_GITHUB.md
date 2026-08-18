# Push checklist

This folder is self-contained for source-code and compact-result publication.
It intentionally excludes large scientific artifacts.

## Review before publishing

```bash
git status --short
git ls-files | grep -E '\.(pt|pth|ckpt|npz|npy|csv|tar|tar\.gz|zip)$' && \
  echo "STOP: large/binary artifact is tracked"
```

The second command should print no tracked artifact and may exit with status 1.

## Create the first commit

```bash
git add .
git status --short
git commit -m "Consolidate ALIGNN descriptor-v41 and coordinate-GPU-v4 experiments"
```

## Connect a GitHub repository

Create an empty repository on GitHub, then use its exact URL:

```bash
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Do not commit credentials, DelftBlue SSH keys, checkpoints, full prediction
tables, or cluster caches. Add the intended license before public release.
