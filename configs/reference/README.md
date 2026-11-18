# Archived training configurations

These files are byte-identical copies of the corresponding development
configurations. Their hashes are part of teacher/cache provenance, so historical
filesystem paths are deliberately retained here. Do not run these files directly
or replace paths in place.

Use `scripts/prepare_training.py` with `configs/paths.example.yaml` to generate
local executable recipes. Generated YAML and its `config_provenance.json` record
the original and resolved hashes separately. The policy uses the archived
teacher YAML for architecture and supplies local checkpoint, normalization, and
vision-encoder paths explicitly.

See [the training guide](../../docs/training.md), including the documented
RoboTwin schedule discrepancy between the supplied paper and development recipes.
