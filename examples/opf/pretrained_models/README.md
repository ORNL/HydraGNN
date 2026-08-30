# OPF pretrained checkpoints

This directory retains the HydraGNN configurations for the pretrained
HeteroSAGE and HeteroHEAT models, but it does not store generated model
checkpoints in Git.

Before running an OPF fine-tuning example, generate or obtain the required
checkpoint and arrange it as

```text
pretrained_models/
  HeteroSAGE_best/
    config.json
    HeteroSAGE_best.pk
  HeteroHEAT_best/
    config.json
    HeteroHEAT_best.pk
```

Pass the parent directory through `--pretrained_model_dir` and select the
corresponding directory with `--pretrained_model_name`. The fine-tuning entry
points validate this path and stop with a clear `FileNotFoundError` when a
checkpoint has not been supplied.

Checkpoint files are deliberately ignored because they are generated training
artifacts. They should be distributed through an artifact store or generated
from the retained configuration, rather than committed to HydraGNN's source
history.
