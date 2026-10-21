# ML4MD: random-noise warm-up for uncertainty calibration

Warm-up experiments on balanced Matbench `mp_is_metal` classification
(metal vs non-metal) for three models.

## Repository structure

| Folder | Purpose | Python |
| --- | --- | --- |
| [crabnet/](crabnet/) | CrabNet composition-based experiments | 3.10 |
| [alignn/](alignn/) | ALIGNN crystal-structure experiments | 3.10 |
| [orb/](orb/) | ORB-v3 crystal-structure experiments, trained from scratch | 3.12 |
| [data/](data/) | Shared dataset splits | — |

Each model folder contains its own README, requirements, experiment runners,
and plotting script. Follow that folder's README and run its commands from
inside the model folder. Use a separate Python environment for each model.

## Shared data

The root `data/` folder contains the balanced dataset splits:

- `train.json`: 2,014 materials.
- `val.json`: 252 materials.
- `test.json`: 252 materials.

Each split contains equal numbers of metallic and non-metallic materials.
The runners locate this folder relative to their source files.

CrabNet and ALIGNN use all three splits, with test evaluation after training.
ORB intentionally uses only training and validation data.

New experiment outputs are saved under each model's `runs/` directory.

## Contributing

- Open an issue before proposing substantial changes.
- Create a branch and submit a pull request describing the changes.
- Keep model-specific code and dependencies inside the relevant model folder.
- Use the Python version and requirements specified for that model.
- Describe the checks performed and report any remaining validation limits.
- Explicitly document changes to dataset splits, seeds, training, or evaluation.
- Keep generated runs, checkpoints, and local environments out of Git.