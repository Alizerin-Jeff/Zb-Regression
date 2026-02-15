# Data Directory

- `raw/`: place immutable source files here (for example, fallback historical CSVs).
- `processed/`: auto-generated cached files from preprocessing/fetch steps.

For the fixed legacy analysis window, the notebook looks for:
- `raw/zb_hourly_2021-12-16_2023-12-16.csv`
- `processed/zb_hourly_2021-12-16_2023-12-16_dubai.csv`
