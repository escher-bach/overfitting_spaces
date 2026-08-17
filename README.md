# Behavioral Geometry of Overfitting

Stage A implements the reproducible execution foundation described in
`EXPERIMENT-PLAN.md`. The retained GPU operator interface is:

```text
python tools/kaggle_run.py validate --experiment preflight
python tools/kaggle_run.py launch --experiment preflight
python tools/kaggle_run.py status --experiment preflight
python tools/kaggle_run.py logs --experiment preflight
python tools/kaggle_run.py collect --experiment preflight
```

`launch` is deliberately non-automatic: it requires a clean, pushed commit,
a configured Kaggle owner and an explicit command. The launcher clones the
pinned source to `/tmp`; `/kaggle/working` is output-only.

The first `preflight` run is allowed to bootstrap the deterministic split
manifest and returns it in the verified audit payload. Before `pilot-overfit`,
promote that manifest to `data/manifests/cifar10-v1.json`, commit it, and replace
the registry's pending hash. The same manifest can instead be generated locally
against torchvision's verified CIFAR-10 download:

```text
python tools/generate_manifest.py --data-root /tmp/cifar10 --download --output data/manifests/cifar10-v1.json
```

Pilot and main runs attach the established `pankrzysiu/cifar10-python` Kaggle
dataset and let torchvision verify its canonical batch checksums. This avoids
paying the slow public-download path in every GPU session.

Do not use `Overfitting.ipynb` as an implementation input; it is historical.
