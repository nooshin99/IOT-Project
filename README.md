# Phase 2: IoT-23 Baseline Detection

This project implements Phase 2 for an IoT anomaly or botnet detection workflow using the IoT-23 dataset.

It covers:

- feature extraction from labeled network-flow data
- three baseline classifiers
- clustering analysis
- feature importance and reduced-feature evaluation

## What It Produces

Running the pipeline generates:

- `outputs/features_preview.csv`
- `outputs/classification_metrics.csv`
- `outputs/confusion_matrix_<model>.png`
- `outputs/clustering_pca.png`
- `outputs/feature_importance.csv`
- `outputs/reduced_feature_metrics.csv`
- `outputs/phase2_summary.txt`

## Supported Input Formats

The script is designed for IoT-23-style labeled flow data and tries to handle:

- `conn.log.labeled`
- CSV files
- TSV files

Expected columns can include common IoT-23 or Zeek flow fields such as:

- `id.orig_h`, `id.resp_h`
- `id.orig_p`, `id.resp_p`
- `proto`, `service`, `conn_state`
- `duration`
- `orig_bytes`, `resp_bytes`
- `orig_pkts`, `resp_pkts`
- `orig_ip_bytes`, `resp_ip_bytes`
- `label`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 src/phase2_iot23.py --data /path/to/conn.log.labeled --output-dir outputs
```

Optional arguments:

```bash
python3 src/phase2_iot23.py \
  --data /path/to/conn.log.labeled \
  --output-dir outputs \
  --sample-size 50000 \
  --test-size 0.2 \
  --random-state 42
```

## Deliverable Mapping

This pipeline directly supports the Phase 2 deliverables:

- one dataset chosen: IoT-23
- one extracted feature table: `features_preview.csv`
- one baseline detection result table: `classification_metrics.csv`
- one clustering result figure: `clustering_pca.png`
- one short summary of useful and removed features: `phase2_summary.txt`
# IOT-Project
