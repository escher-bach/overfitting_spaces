# CIFAR-10 split manifests

The committed generator creates a class-stratified `cifar10-v1.json` after it
checks the local CIFAR-10 labels. It records the exact indices, labels digest,
normalization and its own canonical SHA-256. The generated manifest is then
committed before any retained Kaggle experiment and its digest is copied into
`kaggle/experiments.toml`.
