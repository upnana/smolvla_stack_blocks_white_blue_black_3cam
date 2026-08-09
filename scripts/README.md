# Scripts

Copied from the lab LeRobot checkout used for this experiment.

| File | Purpose |
|------|---------|
| `get-data-stack_3blocks_3cam.sh` | Teleop data collection (3 cameras) |
| `train_smolvla_stack_3blocks_3cam.sh` | SmolVLA training |
| `infer_smolvla_stack_3blocks_3cam.sh` | Offline / robot eval + clip export |

Defaults use absolute paths under `/home/rxn/...`. Override with env vars (`DATA_ROOT`, `OUTPUT_DIR`, camera paths, serial ports) or edit the header section of each script.
