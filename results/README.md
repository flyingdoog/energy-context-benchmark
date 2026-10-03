# Saved experimental results

These are aggregate scores from the completed experiments, not input data,
evaluation labels, model weights, or per-example predictions.

`code/build_paper_assets.py` reads:

- `device_average.csv` and `cu_common_scores.json` for the device table;
- `declarations_tree.csv` and `declarations_nhits.csv` for charging declarations;
- `reassignment_effects.json` for the record-association figure.

The remaining score files retain native-span or supporting aggregates.
`paper_device_cells.json` and `paper_declaration_cells.json` record the numeric
cells used to generate the manuscript tables. Source units, averaging rules,
and eligibility conditions are described in the paper.
