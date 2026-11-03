# Multimodal Tiny RNN for Next-Day Alcohol Craving Prediction

This repository contains the analysis pipeline for predicting next-day alcohol craving from longitudinal ecological momentary assessment (EMA) data and participant-level baseline measurements. The core model is a compact vanilla recurrent neural network (RNN) whose initial hidden state is personalized using static psychological, cognitive, EEG, and heart-rate-variability (HRV) features.

The repository covers the full workflow from preprocessing and exploratory analysis to repeated-seed model evaluation, sliding-window experiments, post-hoc analyses, and modality-specific ablation studies.

## Research overview

The prediction task is formulated as five-class classification of next-day alcohol craving (`tmrw_alcohol_desire`, levels 1–5).

- **Time-varying inputs:** daily EMA and behavioral variables, including depression items, meals, sleep, and alcohol consumption.
- **Static inputs:** participant-level cognitive, alcohol-use, motivational, impulsivity, EEG, and HRV features.
- **Sequence model:** a one-layer `tanh` vanilla RNN.
- **Personalization:** a small feed-forward network maps static features to the RNN's initial hidden state.
- **Evaluation:** accuracy, macro F1, weighted F1, macro precision, macro recall, and mean absolute error (MAE).
- **Robustness:** results are aggregated over 100 runs with seeds 2026–2125.

## Model architecture

For each participant, the model receives a sequence of daily observations and a static baseline vector:

1. The static vector is passed through two linear layers with ReLU, dropout, and a final `tanh` activation.
2. The resulting representation initializes the hidden state of the RNN.
3. The RNN processes the daily sequence.
4. The final hidden state is mapped to five craving classes through a linear classification head.

The primary configuration used in `Best_EPOCH_Multimodal.ipynb` is:

| Setting | Value |
|---|---:|
| Sequence length | 4 days |
| Daily features | 11 |
| Static features | 41 |
| RNN hidden size | 16 |
| RNN layers | 1 |
| Output classes | 5 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Cross-entropy |
| Maximum epochs | 70 |
| Training batch size | 32 |
| Validation/test batch size | 256 |
| Repeated runs | 100 |

For every run, the checkpoint with the lowest validation loss is restored before final train, validation, and test evaluation.

## Temporal data split

Data are sorted by participant and date. Each participant contributes at most 27 daily observations. The split is chronological within each participant:

- training: earlier observations;
- validation: the three days immediately preceding the test period;
- test: the final five days.

Only windows containing consecutive `day_order` values are retained. Labels are converted from levels 1–5 to zero-based class indices during training and converted back for reporting. This design avoids randomly mixing later observations into earlier partitions, although all partitions still contain observations from the same participants.

## Repository structure

```text
.
├── 1. Data Preprocessing/
│   ├── Final_PreprocessBaselineData.ipynb
│   ├── Final_BaselineFeatureSelection.ipynb
│   ├── Final_PreprocessAppData.ipynb
│   ├── Final_PreprocessAppData2.ipynb
│   ├── Final_PreprocessAppData3.ipynb
│   ├── Final_PreprocessAppData4.ipynb
│   └── Final_PreprocessAppData5(FS).ipynb
├── 2.Data Analysis/
│   ├── BaselineDataAnalysis.ipynb
│   ├── AppDataAnalysis.ipynb
│   ├── EMA_Analysis.ipynb
│   └── FigureAnalysis.ipynb
├── 3.Modeling/
│   ├── Best_EPOCH_Unimodal.ipynb
│   ├── Best_EPOCH_Multimodal.ipynb
│   ├── Sliding_Window_2_Best_EPOCH_Unimodal.ipynb
│   ├── Sliding_Window_2_Best_EPOCH_Multimodal.ipynb
│   └── Concatenated_Multimodal.ipynb
├── 4.Post Hoc/
│   ├── Best_Compare_User.ipynb
│   ├── Sliding_Window_2_Best_Compare_User.ipynb
│   ├── Post_hoc_figure_analysis.ipynb
│   └── VarianceAnalysis.ipynb
├── 5.Sliding Windows/
│   ├── 2_100_VanillaRNN_Uni.ipynb
│   ├── 2_100_VanillaRNN_Multi.ipynb
│   ├── 4_100_VanillaRNN_Uni.ipynb
│   ├── 4_100_VanillaRNN_Multi.ipynb
│   ├── 6_100_VanillaRNN_Uni.ipynb
│   ├── 6_100_VanillaRNN_Multi.ipynb
│   ├── 8_100_VanillaRNN_Uni.ipynb
│   └── 8_100_VanillaRNN_Multi.ipynb
└── 6.Ablation Study/
    ├── EEG_Multimodal.ipynb
    ├── HRV_Multimodal.ipynb
    └── PF_Multimodal.ipynb
```

CSV files in these directories contain intermediate datasets, selected feature sets, participant summaries, and post-hoc result tables.

## Main input features

### Daily EMA and behavioral features

The main model uses:

```text
phq1, phq2,
meal_morning, meal_lunch, meal_dinner, meal_night,
sleep_satisfy, sleep_disturb, sleep_minute_adjusted,
amount_beer_glass, amount_soju_glass
```

### Static baseline features

The multimodal baseline vector combines:

- cognitive scores: VCI, PRI, WMI, and IQ;
- alcohol-use measures: alcohol frequency and pure alcohol amount;
- psychological measures: DMQ factors, BIS, and SOCRATES;
- EEG band-power features across selected electrodes;
- HRV/autonomic measures, including autonomic activity, autonomic balance, cardiac stability, Tp, ApEn, Srd, and VLF-related features.

The ablation notebooks evaluate modality-specific static feature sets. `PF_Multimodal.ipynb` retains five psychological-factor features, while `HRV_Multimodal.ipynb` retains eight HRV/autonomic features. `EEG_Multimodal.ipynb` contains the corresponding EEG-focused experiment.

## Installation

The executed notebooks record Python 3.9.23. A compatible environment can be created as follows:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn torch
```

For Apple Silicon or GPU acceleration, install the PyTorch build appropriate for the local platform.

## Recommended execution order

Run notebooks from the repository root or from their own directory while preserving the existing relative paths.

1. Run the notebooks in `1. Data Preprocessing/` to construct the cleaned app/EMA data and selected baseline feature tables.
2. Use `2.Data Analysis/` for descriptive statistics, EMA analyses, and figures.
3. Run the primary unimodal and multimodal experiments in `3.Modeling/`.
4. Run `5.Sliding Windows/` to compare sequence lengths of 2, 4, 6, and 8 days.
5. Run `6.Ablation Study/` for EEG-, HRV-, and psychological-factor experiments.
6. Use `4.Post Hoc/` to compare participant-level results, analyze variance, and generate final figures.

The main modeling notebooks expect these files in the current working directory:

```text
App_FeatureSelection.csv
Baseline_FeatureSelection.csv
```

Because several notebooks use relative paths such as `../4.Post Hoc/...`, changing the working directory can break file loading or result export.

## Recorded results

The stored outputs in the primary four-day multimodal notebook report the following mean ± standard deviation over 100 seeds:

| Split | Accuracy | Macro F1 | Weighted F1 | MAE |
|---|---:|---:|---:|---:|
| Train | 0.5864 ± 0.0228 | 0.4260 ± 0.0592 | 0.5327 ± 0.0397 | 0.6148 ± 0.0390 |
| Validation | 0.5254 ± 0.0249 | 0.3208 ± 0.0466 | 0.4860 ± 0.0312 | 0.7163 ± 0.0401 |
| Test | 0.5558 ± 0.0213 | 0.3723 ± 0.0447 | 0.5102 ± 0.0335 | 0.6790 ± 0.0354 |

The 95% confidence interval for multimodal test accuracy is 0.5516–0.5600.

Modality-specific static feature experiments stored in the ablation notebooks report:

| Static feature set | Test accuracy | Test macro F1 | Test weighted F1 | Test MAE |
|---|---:|---:|---:|---:|
| HRV/autonomic | 0.5291 ± 0.0212 | 0.3402 ± 0.0442 | 0.4766 ± 0.0290 | 0.7298 ± 0.0373 |
| Psychological factors | 0.5005 ± 0.0231 | 0.3277 ± 0.0428 | 0.4498 ± 0.0305 | 0.7963 ± 0.0410 |

These values are notebook outputs, not results recomputed during installation. Re-running the full 100-seed experiments may take substantial time.

## Reproducibility notes

- Seeds are applied to Python, NumPy, and PyTorch.
- The notebooks use CPU tensors directly and do not currently include explicit CUDA/MPS device handling.
- PyTorch deterministic-algorithm settings are not enabled, so exact results may vary across platforms or library versions.
- The training loader is shuffled, while validation and test loaders are not.
- Class imbalance is visible in the stored confusion matrix; accuracy should therefore be interpreted together with macro F1 and MAE.
- One baseline record is removed by positional index before `user_order` is regenerated. Verify this step if the upstream table changes.
- Some result-export lines are commented out and must be enabled when regenerating downstream post-hoc CSV files.

## Data availability and privacy

This project includes human-participant research data and participant identifiers. Do not publish raw or row-level CSV files unless data sharing is explicitly permitted by the study protocol, institutional review board, and participant consent.

For a public GitHub repository, consider tracking only code, documentation, schemas, and approved aggregate results. Keep restricted data outside Git and provide access instructions or synthetic examples instead.

Suggested `.gitignore` entries:

```gitignore
.DS_Store
.ipynb_checkpoints/
.venv/
__pycache__/

# Review before enabling: raw or participant-level research data
*.csv
```

If selected aggregate CSV files are safe to publish, replace the broad CSV rule with directory- or filename-specific rules.

## Citation

If this repository supports a paper or thesis, add the final citation and DOI here once available:

```bibtex
@article{citation_pending,
  title   = {Multimodal Prediction of Alcohol Craving Dynamics with a Tiny RNN},
  author  = {Author information pending},
  journal = {Publication information pending},
  year    = {Year pending}
}
```

## License

No license file is currently included. Add an explicit software license before public redistribution. Note that a code license does not grant permission to redistribute participant data.
