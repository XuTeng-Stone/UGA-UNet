# Legacy experiment code

These are earlier BUSI/BrEaST experiment scripts from the original workspace.
They are superseded by the cleaned [`ugaunet/`](../../ugaunet/) package and are
kept only for provenance.

- Root scripts are flat and import each other by module name.
- `BUSI/BUSI_code/` and `BrEaST/BrEaST_code/` contain the older dataset-specific
  training/anchor scripts.

They expect the original absolute data paths and are not maintained.  Use the
top-level `scripts/train.py` for new runs.
