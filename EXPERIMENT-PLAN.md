# Behavioral Geometry of Overfitting

## Image-classification experiment and Kaggle execution plan

### Status and scope

This document is the working plan for the first principled experiment. Its
purpose is to determine whether overfitting leaves a coherent, measurable, and
reproducible signature in a model's logits on a fixed unlabeled probe set.

The existing `Overfitting.ipynb` is historical context only. It is not evidence
for the current hypothesis, and its use of CIFAR-10, endpoint comparisons, MMD,
or Gaussian kernels does not constrain this design. The current experiment is
specified independently below.

Version 1 is deliberately limited to image classification. It does not claim
that any discovered structure is universal across architectures, datasets, or
modalities. Cross-domain replication is a later stage.

---

## 1. Research question

For a checkpoint \(M_{i,t}\) from independent training run \(i\) at time \(t\),
let

\[
\phi(M_{i,t}) = [z_{i,t}(x_1),\ldots,z_{i,t}(x_P)]
\]

be its logits on the same fixed probe images. Probe labels are not used.

The primary question is:

> Can the amount by which held-out generalization has deteriorated be predicted
> from \(\phi(M_{i,t})\) on a completely unseen training run, beyond what is
> explained by training time, confidence, entropy, and ordinary performance?

The experiment separates three claims:

1. **Detectability:** behavioral state predicts the degree of overfitting on
   unseen runs.
2. **Geometric coherence:** checkpoints with similar overfitting state are close
   under a reproducible metric after nuisance controls.
3. **Trajectory coherence:** independent runs move in aligned behavioral
   directions as generalization begins to deteriorate.

Detectability is primary. Geometry is secondary. Trajectory and future-warning
analyses are exploratory until the first two claims survive confirmation.

---

## 2. What counts as overfitting

The primary generalization measurement is evaluation cross-entropy, not
accuracy. Loss is sensitive to worsening calibration and confident mistakes
even when accuracy is temporarily unchanged.

For each run, define deterioration at checkpoint \(t\) as

\[
g_{i,t} = L^{\mathrm{eval}}_{i,t}
          - \min_{s \leq t} L^{\mathrm{eval}}_{i,s}.
\]

Thus \(g=0\) while a run is at its best observed generalization and \(g>0\)
after its evaluation loss has worsened. This is different from the
train-evaluation gap, which can grow merely because training loss improves.

Secondary outcomes are evaluation accuracy deterioration, the generalization
gap, and calibration error. They do not replace the primary target after
results are observed.

Checkpoints are evaluated on a fixed cadence. The cadence is not adapted to
where a particular run appears to peak.

---

## 3. Experimental system

### 3.1 Dataset and fixed partitions

Use the maintained `torchvision.datasets.CIFAR10` dataset. Create one
class-stratified split manifest with an explicitly recorded split seed:

| Partition | Source | Size | Role |
|---|---:|---:|---|
| training | CIFAR-10 train | 10,000 | receives gradients |
| probe | CIFAR-10 train | 5,000 | fixed inputs; labels ignored |
| recipe validation | CIFAR-10 train | 5,000 | pilot-only recipe selection |
| unused reserve | CIFAR-10 train | 30,000 | untouched in version 1 |
| final evaluation | CIFAR-10 test | 10,000 | defines \(g\) in main runs |

The manifest records the exact indices, class counts, raw-data identity,
normalization, and SHA-256. All runs use the same training, probe, and
evaluation examples. This isolates optimization-run variability. A later
robustness stage can repeat the experiment with new training-data splits.

The representation extractor accepts probe images without exposing their
labels to training or analysis code. A contract test verifies this boundary.

### 3.2 Model

Use `torchvision.models.resnet18` with the conventional CIFAR stem adaptation:
a 3x3 stride-1 first convolution and no max-pool. This is the only
project-owned architectural adjustment. Do not create a custom ResNet.

Every main run uses the same architecture and initial-weight distribution.
Runs differ only in their declared root seed, which deterministically derives
initialization, data-order, and worker seeds.

### 3.3 Initial training recipe

The pilot begins with a deliberately overfit-capable but ordinary recipe:

- SGD with momentum;
- cross-entropy loss;
- no label noise;
- no data augmentation;
- no weight decay;
- default batch size 512, subject to Kaggle preflight validation of memory use
  and achieved throughput;
- a declared learning-rate schedule;
- 200 epochs;
- metrics every epoch;
- probe logits and evaluation metrics every 5 epochs, including epoch 0.

Exact optimizer and scheduler values live in a versioned configuration file,
not in a notebook. The purpose of the pilot is to verify that this recipe
produces a clear rise in held-out loss after an earlier minimum on multiple
seeds. The selected recipe, including the validated batch size and all
performance settings, is fixed once pilot training starts.

If it does not, changes are tried in this predeclared order using only the
recipe-validation split:

1. extend training duration;
2. reduce the training subset from 10,000 to 5,000 examples;
3. introduce a small, fixed amount of training-label noise.

Stop at the first recipe that reliably produces the required trajectory. Do
not tune the recipe on the final evaluation set. Freeze the selected recipe,
checkpoint cadence, split manifest, and analysis protocol before main runs.

### 3.4 Pilot gate

Run two pilot seeds concurrently, one per T4. The main experiment may begin
only if both show:

- a well-defined validation-loss minimum before the final checkpoint;
- sustained deterioration after that minimum rather than a one-checkpoint
  fluctuation; and
- continued improvement or saturation of training loss during the same
  interval.

The pilot establishes that the system actually overfits. It does not test the
behavioral hypothesis and is excluded from confirmatory results.

---

## 4. Main runs and independence

The target main sample is 20 independent optimization runs:

- 12 development runs for representation choices, nested run-level
  cross-validation, and model fitting;
- 8 locked confirmation runs evaluated once after the protocol is frozen.

If compute constraints require a smaller sample, the revised run count must be
recorded before confirmation outputs are examined; 12 total main runs is the
minimum acceptable exploratory study. Checkpoints from one run are correlated
measurements, not independent samples. Every split, bootstrap, permutation, and
uncertainty calculation therefore treats the complete training run as the
independent unit.

The fixed seed list and its development/confirmation assignment are committed
before launching main runs. Failed infrastructure runs may be rerun with the
same seed only when the failure occurred before scientifically relevant state
was produced; otherwise both attempts remain in the audit record.

---

## 5. Recorded data

For every checkpoint cadence point, record:

- epoch and optimizer step;
- training loss and accuracy;
- evaluation loss, accuracy, and calibration statistics;
- probe logits in fixed probe order;
- mean logit norm, predictive entropy, maximum probability, and predicted
  class histogram;
- wall-clock timing and observed device information.

Store logits as float32 tensors in `safetensors`, with JSON metadata containing
their exact shape, probe-manifest hash, checkpoint identity, and tensor hash.
Store scalar and tabular metrics in Parquet or JSON Lines. Avoid pickle-based
research artifacts.

Full model weights are not part of the behavioral representation. Kaggle may
retain the minimum checkpoints needed for failure recovery or later declared
follow-up analyses, but routine local collection contains logits, metrics,
reports, and plots rather than checkpoint weights.

---

## 6. Behavioral representations

Evaluate a small predeclared family rather than searching freely:

1. **Raw logits:** the direct \(P \times C\) output tensor.
2. **Class-centered logits:** subtract the mean across classes for each probe,
   removing the arbitrary additive logit offset.
3. **Scale-normalized logits:** class-center and divide by a declared per-probe
   norm with a numerical epsilon, reducing confidence-magnitude effects.
4. **Probabilities:** softmax outputs, included as a bounded comparison.

All preprocessing, dimensionality reduction, and prediction models are fit on
development runs only. Confirmation runs never influence centering statistics,
PCA components, regularization strength, or representation selection.

Gaussian-kernel MMD is not part of version 1. It may be added later only if a
specific failure of linear or ordinary metric methods justifies it, with a
bandwidth fixed from development data and an estimator appropriate for the
number of independent model runs.

---

## 7. Primary analysis: held-out-run detectability

Use maintained scikit-learn components for the initial detector:

1. optional development-fit standardization;
2. randomized PCA fit only on training runs;
3. ridge regression predicting \(g_{i,t}\).

Hyperparameters are selected by nested cross-validation that leaves complete
runs out. No checkpoint from a held-out run may appear in any fitting or model
selection step.

Compare the behavioral detector against predeclared baselines:

- epoch or optimizer step alone;
- training loss and accuracy;
- probe logit norm, entropy, maximum probability, and class histogram;
- all preceding unlabeled nuisance features together;
- evaluation accuracy as a labeled analysis-only control.

The primary endpoint is the improvement in checkpoint-level prediction error
from adding the behavioral representation to the nuisance baseline, evaluated
on confirmation runs. Also report out-of-run \(R^2\), MAE, and Spearman
correlation, both pooled and separately for every held-out run.

Uncertainty is obtained by resampling whole runs. A permutation control shuffles
run-level trajectory identities without breaking the checkpoint ordering
within a run.

Evidence for detectability requires all of the following:

- behavior improves over the nuisance baseline on locked confirmation runs;
- the effect is not driven by one run;
- centered or normalized representations retain useful signal if raw logits
  win mainly through magnitude; and
- results are reported alongside the epoch-only and confidence-only controls.

Failure is informative. If nuisance variables explain all predictive power,
the conclusion is that this experiment did not find a distinct behavioral
signature of overfitting.

---

## 8. Secondary geometry and trajectory analyses

### 8.1 Geometric coherence

For representations selected on development data:

- compute Euclidean and cosine distances between checkpoints from different
  runs;
- test association with \(|g_{i,t}-g_{j,s}|\);
- repeat within matched epoch bands and matched evaluation-accuracy bands;
- compare within-state distances against between-state distances; and
- estimate uncertainty by resampling runs, never probe images as if they were
  independent models.

A large distance from a generalizing centroid is not sufficient. Overfit
checkpoints must also show reproducible within-state coherence relative to
cross-state distance.

### 8.2 Direction of motion

For fixed checkpoint intervals, compute

\[
\Delta\phi_{i,t}=\phi(M_{i,t+\Delta t})-\phi(M_{i,t}).
\]

Measure cross-run cosine alignment at matched generalization states and test
whether \(\Delta\phi\) predicts subsequent \(\Delta g\). Any claim of early
warning requires predicting future deterioration before the evaluation target
used to define it has worsened, on locked runs. This is exploratory in version
1 and must not be promoted to the primary result after inspection.

---

## 9. Kaggle execution architecture

The reusable workflow from `baby-llm-foundations` is adopted as an operational
pattern, not as research code. This repository gets its own configurations,
runner, artifacts, and audit schema.

### 9.1 Principles retained from the reference workflow

- The official Kaggle CLI owns Kaggle API operations; project code is only a
  thin adapter.
- A Kaggle notebook is a launcher, not the experiment implementation.
- Each retained run is tied to an exact Git commit, canonical configuration
  hash, seed set, observed hardware allocation, and immutable Kaggle notebook
  version.
- Long jobs use `launch`, followed by independent `status`, `logs`, and
  `collect` operations. A lost local shell must not lose the remote run.
- Source clones and caches live under `/tmp`; `/kaggle/working` contains only
  declared result artifacts.
- CIFAR-10 is attached as the declared `pankrzysiu/cifar10-python` Kaggle
  dataset source after the bootstrap preflight; torchvision verifies the
  canonical batch checksums before use.
- Successful and failed runs both publish bounded diagnostics and checksums.
- Routine collection downloads compact scientific/audit artifacts, not the
  full Kaggle output tree or heavyweight checkpoints.
- No tool pushes Git, launches a GPU run automatically, or reuses an accepted
  artifact slug for different content.
- Every GPU launch requires explicit user authorization.

### 9.2 Sole operator interface

Provide one repository command:

```text
python tools/kaggle_run.py validate --experiment <name>
python tools/kaggle_run.py launch   --experiment <name>
python tools/kaggle_run.py status   --experiment <name>
python tools/kaggle_run.py logs     --experiment <name>
python tools/kaggle_run.py collect  --experiment <name>
```

An optional `run` command may perform validate, launch, wait, and collect for a
short smoke test. Long training runs should use the non-blocking sequence.

The adapter shells out to pinned, documented `kaggle kernels` commands. It owns
only repository-specific validation, ephemeral staging, notebook rendering,
provenance, checksum verification, and audit-receipt generation.

Before submission it must verify:

- the experiment exists in the declarative registry;
- the working tree paths relevant to the experiment are clean;
- the exact commit is reachable from the configured remote;
- the config and data-split manifest hashes match the registry;
- the Kaggle credentials and CLI are available;
- no conflicting submission is targeting the same mutable kernel; and
- requested acceleration is explicit.

The Kaggle-side preflight records the actual devices and fails before training
if they do not match the experiment contract. Hardware observed in the run is
evidence; the requested accelerator string is not.

### 9.3 Launcher notebook contract

The rendered notebook contains only:

1. immutable launch constants;
2. clone and checkout of the exact Git commit into `/tmp`;
3. pinned dependency installation;
4. one invocation of the repository-owned experiment runner; and
5. failure propagation after diagnostic packaging.

It contains no model definition, training loop, dataset logic, analysis, or
manual patch. Fixes happen in Git and produce a new immutable submission.

### 9.4 Kaggle experiment registry

Use `kaggle/experiments.toml` to declare experiment name, purpose, config,
split-manifest identity, seed set, slug prefix, output contract, and accelerator
request. Published experiment entries are immutable; changed science or code
gets a new entry and slug.

Initial entries should be:

- `preflight`: environment, imports, dataset identity, one minibatch, one probe
  pass, artifact packaging, and device check;
- `pilot-overfit`: two concurrent validation-only seeds;
- `main-development-*`: bounded seed batches from the development set;
- `main-confirmation-*`: locked seed batches submitted only after the analysis
  contract is frozen;
- `analysis-confirmation`: consumes verified upstream logits and metrics,
  trains nothing, and produces the final report.

Independent seeds can run one model per visible GPU when the observed
allocation supports it. This is preferable to distributed training of one
small ResNet because independent trajectories are the scarce scientific unit.
For the expected two-T4 allocation, run two independent seeds concurrently,
one per GPU, and do not use DDP. The preflight must benchmark and verify the
chosen scheduling arrangement.

### 9.5 Output and audit contract

Each Kaggle run writes beneath one run-specific directory:

```text
/kaggle/working/overfitting-results/<run-id>/
  run_manifest.json
  resolved_config.json
  phase_status.json
  environment.json
  logs/
  data_manifest.json
  metrics/
  logits/
  analysis/
    summary.json
    result-report.json
    plots/
  recovery/
  SUCCESS                 # or FAILURE
```

Package two logical artifacts with SHA-256 sidecars:

1. **Analysis artifact:** provenance, configs, split identity, phase status,
   metrics, logits needed for declared analysis, compact logs, results, and
   plots.
2. **Recovery artifact:** minimum checkpoint and optimizer state required to
   resume or support a declared downstream run.

Recovery artifacts stay on Kaggle by default. If raw trajectory logits make
the analysis artifact too large for routine collection, retain them as a
separate verified tensor artifact and collect only derived tables plus the
subset required to reproduce headline analyses. Do not silently discard raw
logits.

After collection, write a tracked receipt containing:

- schema version and run ID;
- experiment name and purpose;
- exact Kaggle owner, slug, version, URL, and terminal status;
- source remote and full Git SHA;
- canonical config and split-manifest SHA-256 values;
- root seeds and observed device inventory;
- artifact paths, sizes, and SHA-256 values;
- upstream run identities, if any; and
- scientific report path.

A Kaggle `COMPLETE` status proves only that the notebook ended. Results may be
used for a claim only after their artifact checksums, identities, phase status,
and receipt are verified.

---

## 10. Planned repository structure

The current directory is not yet a Git repository. Git initialization and a
configured remote are prerequisites for commit-pinned Kaggle automation.

```text
overfitting_spaces/
  EXPERIMENT-PLAN.md
  README.md
  pyproject.toml
  requirements-kaggle.txt
  configs/
    pilot.toml
    main.toml
    analysis.toml
  data/
    manifests/
  kaggle/
    experiments.toml
    launcher-template.ipynb
  src/overfitting_spaces/
    data.py
    model.py
    train.py
    representations.py
    analysis.py
    artifacts.py
    runner.py
  tools/
    kaggle_run.py
  tests/
    test_splits.py
    test_probe_label_boundary.py
    test_determinism.py
    test_artifact_contract.py
    test_run_isolation.py
    test_kaggle_control_plane.py
  audit/runs/
  reports/
  legacy/
    Overfitting.ipynb
```

This is a target layout, not authorization to move the historical notebook
yet. Standard libraries own training, models, metrics, tensor serialization,
configuration parsing, and regression. Project code is limited to the
experimental protocol and thin orchestration adapters.

---

## 11. Implementation and execution stages

### Stage A — Local foundation

1. Initialize Git and configure the remote.
2. Establish the Python package, pinned Kaggle compatibility dependencies, and
   tests.
3. Generate and commit the deterministic data manifest.
4. Implement one local CPU minibatch smoke test and artifact round trip.
5. Implement the Kaggle registry, launcher renderer, and CLI adapter.

**Gate:** local tests pass; a rendered notebook contains no experiment logic;
`validate` performs no remote mutation.

### Stage B — Kaggle preflight

Run a fresh CLI-submitted batch that checks environment, actual GPUs, dataset
identity, deterministic seeding, one training step, one evaluation pass, one
probe-logit export, failure packaging, and collection.

**Gate:** verified receipt and artifact round trip. No scientific claim.

### Stage C — Overfitting pilot

Run two concurrent pilot seeds against recipe-validation data. Apply the predeclared
fallback order only if the pilot gate fails. Freeze the successful recipe and
analysis protocol.

**Gate:** documented evidence of genuine within-run deterioration and a frozen
main-run specification.

### Stage D — Main development runs

Launch the 12 development seeds in immutable bounded batches. Collect and
verify each batch before it is admitted to analysis. Develop representations
and detectors using run-level cross-validation only.

**Gate:** one frozen representation-selection and detector procedure, with all
nuisance baselines defined.

### Stage E — Locked confirmation

Launch the eight confirmation seeds. Run the frozen analysis once. Any later
changes are labeled exploratory and do not replace the original confirmation
result.

**Gate:** verified receipts, complete run inventory, final machine-readable
result report, plots, and prose interpretation including negative outcomes.

### Stage F — Robustness, only if justified

Repeat the frozen protocol on a new training-data split, then another image
dataset or architecture. A later sequence-domain replication is required before
making modality-general claims.

---

## 12. Acceptance criteria for version 1

The experiment is complete when:

1. genuine evaluation-loss deterioration occurs within matched training runs;
2. every analyzed checkpoint is linked to an exact run, config, seed, data
   manifest, commit, and verified Kaggle version;
3. probe labels never enter representation construction or detector fitting;
4. all model selection is confined to development runs;
5. confirmation runs are evaluated by the frozen procedure exactly once;
6. epoch, confidence, entropy, training metrics, and ordinary-performance
   controls are reported;
7. uncertainty treats runs—not checkpoints or probe images—as independent;
8. successful and failed remote runs leave auditable artifacts;
9. the result is stated at the strength supported by the evidence; and
10. a null result remains a valid completed outcome.

---

## 13. Deliberate non-goals

Version 1 will not include:

- Gaussian-kernel bandwidth searches or MMD claims;
- weight-space geometry;
- architecture or dataset sweeps;
- learned nonlinear manifold models;
- intervention, regularization, or negative distillation;
- automatic Kaggle launches on Git push;
- a custom Kaggle API client, distributed scheduler, or experiment dashboard;
- manual notebook patching as part of the retained workflow; or
- universal claims about overfitting across modalities.

The first result should answer one narrow question cleanly before the project
adds more geometry or more domains.
