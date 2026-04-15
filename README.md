# vision-bias-steering

Code for spatial-versus-descriptive steering experiments in language-model outputs.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Main Entry Points

- `bias_steering/run.py`: train, validate, and evaluate steering vectors
- `run_local_sweep.py`: local caption sweep and plotting workflow
- `run_multimodel_sweep.py`: multi-model sweep runner
- `plotting/master_prompt_experiments.py`: prompt-analysis plotting entry point

## Repository Layout

```text
vision-bias-steering/
├── bias_steering/
│   ├── run.py
│   ├── config.py
│   ├── data/
│   ├── steering/
│   └── eval/
├── experiments/
│   ├── coherence_frontier/
│   ├── prompt_template_search/
│   └── rivanna/
├── plotting/
├── runs_vision/
├── run_local_sweep.py
├── run_multimodel_sweep.py
├── test_multimodel_sweep.py
└── test_prompt_steering.py
```

## Notes

- Generated plots, paper files, local result bundles, and local tool config are ignored.
- Keep committed changes focused on source code, tests, small configs, and essential dataset metadata.

## License

See `LICENSE`.
