# Development data access

This document is the access boundary for analysis of the completed CIFAR-10
development runs. It was written before development-result analysis. The six
successful runs are fixed at Git commit
`b42d5e48d17d122e9da82640194464c3dc5af902`.

## Dataset and split identity

The source dataset is `torchvision.datasets.CIFAR10.train`, with the standard
CIFAR-10 test set used as the main-run evaluation set. The project registry
declares Kaggle dataset source `pankrzysiu/cifar10-python`; each successful run
recorded the actual fallback resolution `tmp_download` in its provenance. In
both cases torchvision performs its canonical CIFAR batch validation.

The committed split contract is
[`data/manifests/cifar10-v1.json`](data/manifests/cifar10-v1.json):

- split seed: `1729`
- manifest SHA-256:
  `c99c82659dcae6c874e23ade21fb1e6a3ee75202e2ab0c711c545d30e7f1efdb`
- class-stratified partitions of the CIFAR-10 training set: 10,000 training,
  5,000 fixed unlabeled probe, 5,000 recipe-validation, and 30,000 reserve
- normalization mean `[0.4914, 0.4822, 0.4465]`; standard deviation
  `[0.247, 0.2435, 0.2616]`

Probe labels are not an analysis input. The recipe-validation split was only
for the pilot; the development runs use the official CIFAR-10 test set for the
recorded evaluation target.

## Local, already-collected audit data

The compact, verified material is in `audit/runs/<run-id>/`:

- `receipt.json`: immutable Kaggle identity, artifact hashes, config and
  manifest identities, and collection status.
- `provenance.json`, `run_manifest.json`, `resolved_config.json`,
  `phase_status.json`, and `environment.json`: execution provenance.
- `seeds/seed-<root-seed>/metrics/history.json`: per-epoch training metrics
  plus checkpoint metrics and derived nuisance summaries. These are the local
  inputs for metric-only development analysis.

The raw fixed-probe logit tensors are deliberately not in the compact local
collection. They remain in the verified **recovery archive** of each exact
Kaggle output. Within a recovery archive they are at:

```text
<run-id>/seeds/seed-<root-seed>/logits/epoch-<NNN>.safetensors
```

Each tensor is float32 and its corresponding checkpoint record in the seed
history contains its relative path, SHA-256, and shape. The archive also
contains recovery material; do not treat model checkpoints as the behavioral
representation.

## Authorized development-run inventory

All rows below have `success: true`, terminal status `COMPLETE`, two seed
metric histories, and a completed data-resolution and training phase in their
local audit record. The recovery SHA-256 is the expected checksum for the
archive that contains raw logits.

| Experiment | Run ID | Exact Kaggle version | Root seeds | Recovery archive SHA-256 |
|---|---|---|---|---|
| `main-development-01` | `run-ef7523a1358a` | `aniruddhavarma/overfit-main-development-01-b42d5e4/1` | `881886696`, `1226023601` | `ee5e4be6d2045e2bbfab0c327114382ba5456058df2f4a82931b146f21adc709` |
| `main-development-02` | `run-be83c9d3e5af` | `aniruddhavarma/overfit-main-development-02-b42d5e4/1` | `3549333581`, `2325060526` | `d71663b225a92f30621cda29f42929215f520390d93bf5c2890e966b850adf50` |
| `main-development-03` | `run-0e25bbe73f49` | `aniruddhavarma/overfit-main-development-03-b42d5e4/1` | `4247511459`, `3127276340` | `1eac0d9d4874f10036d2b695415475dc3d5581cfa0834d81f2e5ea9f9bb62036` |
| `main-development-04` | `run-ebbe4035ec6e` | `aniruddhavarma/overfit-main-development-04-b42d5e4/1` | `3002361271`, `1741336945` | `12105310e5028ae7e2c8cc00bd9231d41e71adce7e20eb15ac3d4716d16ac247` |
| `main-development-05` | `run-f856f0c1ae51` | `aniruddhavarma/overfit-main-development-05-b42d5e4/1` | `310190245`, `2722409734` | `cd8d0b1e71ce242447c8fccc847f90776e845a65b25248df17c28772b8c020e1` |
| `main-development-06` | `run-98357fef99fe` | `aniruddhavarma/overfit-main-development-06-b42d5e4/1` | `4292514521`, `971400669` | `df802af8badd3f668c4b567b2145f2e90d83b47b9e8ad693a29005f820982012` |

The receipts are the source of truth for each row:
`audit/runs/<run-id>/receipt.json`. They also identify the compact analysis
archive and its checksum. Do not substitute the failed pre-fix
`main-development-01-7e42407` run (`run-1d017ace48f4`) for development data.

## Retrieve and verify raw logits

Use the official Kaggle CLI only. Keep downloaded archives and extractions
outside the tracked source tree; for example:
`C:\Users\Aniruddha\.cache\overfitting-spaces\development`.

The following PowerShell sequence retrieves one recovery archive, verifies it
against its local receipt, checks the member namespace before extraction, and
extracts into that external cache. Substitute a row from the inventory only as
a matched `(run, exact version)` pair.

```powershell
$repo = 'C:\Users\Aniruddha\Documents\ani\overfitting_spaces'
$cache = 'C:\Users\Aniruddha\.cache\overfitting-spaces\development'
$run = 'run-ef7523a1358a'
$exactVersion = 'aniruddhavarma/overfit-main-development-01-b42d5e4/1'
$receipt = Get-Content -Raw -LiteralPath "$repo\audit\runs\$run\receipt.json" | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $cache | Out-Null
kaggle kernels output $exactVersion -p $cache --file-pattern $receipt.recovery_artifact.path -o
$archive = Join-Path $cache $receipt.recovery_artifact.path
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $receipt.recovery_artifact.sha256) { throw 'Recovery archive SHA-256 mismatch.' }
$members = tar -tzf $archive
if ($members | Where-Object { $_ -notlike "$run/*" }) { throw 'Unexpected archive member path.' }
$destination = Join-Path $cache "extracted\$run"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
tar -xzf $archive -C $destination
```

For a compact-artifact re-download, replace the final retrieval target with
`$receipt.analysis_artifact.name`, verify it against
`$receipt.analysis_artifact.sha256`, and do not expect raw logits in that
archive. The repository adapter's normal compact collection command is:

```powershell
py -3.13 tools\kaggle_run.py collect --experiment main-development-01 --commit b42d5e48d17d122e9da82640194464c3dc5af902
```

It intentionally excludes recovery archives, so it cannot replace the direct
Kaggle CLI retrieval above when raw logits are required.

## Development-analysis boundary

Permitted development inputs are exactly the twelve seed trajectories above:
their local metric histories, verified raw probe logits after the recovery
retrieval procedure, and the associated manifest/config/provenance. Fit every
representation transform, dimensionality reduction, hyperparameter choice, and
detector using development runs only, with complete runs as the independent
unit and with held-out-run validation.

Do **not** retrieve, inspect, extract, attach, or otherwise access any
`main-confirmation-*` output or artifact before the development procedure is
frozen. Confirmation outputs must not influence representation choice,
centering/scaling statistics, PCA, hyperparameters, baselines, plots, or
interpretation during this stage.
