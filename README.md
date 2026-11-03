# Forecasting Next-Day Alcohol Craving in Individuals with Problematic Alcohol Use Drinkers With Multimodal Data and a Tiny Recurrent Neural Network: Prospective Longitudinal Study

This repository provides the research code accompanying our accepted paper on multimodal next-day alcohol craving prediction. It contains the analysis pipeline for predicting next-day alcohol craving from longitudinal ecological momentary assessment (EMA) data and participant-level baseline measurements. The core model is a compact vanilla recurrent neural network (RNN) whose initial hidden state is personalized using static psychological, cognitive, EEG, and heart-rate-variability (HRV) features.

The repository covers the full workflow from preprocessing and exploratory analysis to repeated-seed model evaluation, sliding-window experiments, post-hoc analyses, and modality-specific ablation studies.

> **Data access:** Clinical and participant-level CSV data are not publicly available. Please contact the corresponding author to inquire about access; see [Data availability and contact](#data-availability-and-contact). The notebooks require these restricted inputs to run.

<!-- Before release: verify the supplied title wording "Alcohol UseDrinkers w With" and add authors, journal, DOI, and corresponding author email. -->

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
│   └── Sliding_Window_2_Best_EPOCH_Multimodal.ipynb
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

The tree shows the notebooks intended for the public release. Clinical CSV files, intermediate tables, and participant-level result tables are excluded. `3.Modeling/Concatenated_Multimodal.ipynb` is also excluded from the public release.

| Directory | Purpose |
|---|---|
| `1. Data Preprocessing/` | Clean app records, derive sleep/alcohol variables, construct shifted targets, standardize inputs, and select features. |
| `2.Data Analysis/` | Summarize demographics, EMA adherence, craving distributions, and participant-level variability/entropy. |
| `3.Modeling/` | Compare unimodal and multimodal RNNs with validation-selected checkpoints using four-day and two-day windows. |
| `4.Post Hoc/` | Compare participant-level accuracy using paired tests and explore correlations with craving characteristics. |
| `5.Sliding Windows/` | Evaluate 2-, 4-, 6-, and 8-day sequences for both model variants across 100 seeds. |
| `6.Ablation Study/` | Combine the same daily EMA inputs with EEG-only, HRV-only, or psychological-factor-only static inputs. |

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

The ablation notebooks evaluate modality-specific static feature sets. `PF_Multimodal.ipynb` retains five psychological-factor features, while `HRV_Multimodal.ipynb` retains eight HRV/autonomic features. `EEG_Multimodal.ipynb` retains 22 EEG features. These experiments retain one static modality at a time; all include daily EMA inputs.

## Installation

The notebook metadata records Python 3.9.23. The following commands install the packages imported by the notebooks; dependency versions are not pinned, so this is a setup guide rather than an exact reconstruction of the original environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn torch
```

For Apple Silicon or GPU acceleration, install the PyTorch build appropriate for the local platform.

## Recommended execution order

Start Jupyter from the repository root, then open notebooks in their respective directories. Each kernel must use the directory containing its notebook as the working directory so that local CSV reads and `../4.Post Hoc/` exports resolve correctly.

```bash
jupyter notebook
```

Obtain authorized data access before running data-dependent cells. The public code alone cannot reproduce the experiments without the restricted CSV inputs.

1. In `1. Data Preprocessing/`, run `Final_PreprocessAppData.ipynb`, then `Final_PreprocessAppData2.ipynb`, `Final_PreprocessAppData3.ipynb`, `Final_PreprocessAppData4.ipynb`, and `Final_PreprocessAppData5(FS).ipynb`. Run `Final_BaselineFeatureSelection.ipynb` after `RAW_LAG_APP.csv` and the externally prepared `RAW_Scaled_Baseline.csv` are available.
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

## Required data and generated outputs

All CSV files below are restricted and are omitted from the public release.

| Stage | Main inputs | Outputs / use |
|---|---|---|
| App preprocessing | `records_record.csv`, `records_record (1).csv` | `PrepApp1.csv` → `PrepApp2.csv` → `RAW_APP_NO_LAG.csv` / `RAW_LAG_APP.csv` → `APP_Scaled_LAG.csv` → `App_FeatureSelection.csv` |
| Baseline feature selection | `RAW_Scaled_Baseline.csv`, `RAW_LAG_APP.csv` | Currently writes `Baseline_FeatureSelection copy copy.csv`; verify the selected columns and prepare the `Baseline_FeatureSelection.csv` expected by modeling. |
| Descriptive analyses | `PrepBaseline3.csv`, `APP_Scaled_LAG.csv`, `RAW_APP_NO_LAG.csv` | Descriptive statistics, figures, and participant summaries. |
| Unimodal modeling | `App_FeatureSelection.csv` | Run-level metrics, predictions, and participant accuracy summaries. |
| Multimodal modeling and ablation | `App_FeatureSelection.csv`, `Baseline_FeatureSelection.csv` | Run-level metrics, predictions, and participant accuracy summaries. |
| Post-hoc analyses | `Best_df_user_UNI.csv`, `Best_df_user_MULTI.csv`, corresponding `Sliding_window_2_...` files, `Participant_summary.csv`, `Participant_valid_sample_summary.csv`, and `App_FeatureSelection.csv` as needed | Paired Wilcoxon / t-tests, correlation analyses, and figures. |

Place authorized copies of each input in the directory of the notebook that reads it. Baseline preprocessing before `RAW_Scaled_Baseline.csv` and `PrepBaseline3.csv` is not implemented in the notebooks included here.

The selected app table contains participant/date fields (`user_id`, `user_order`, `date`, `day_order`, `date_diff`), the 11 daily features listed above, and `tmrw_alcohol_desire`. Multimodal notebooks use the 41 explicitly listed baseline features and build a participant-to-static-vector mapping. Preserve the original column names, including Korean names, and participant alignment.

Model summaries are held in notebook variables such as `df_runs`, `df_hist`, and `df_user_summary`. Some participant-level export statements are commented out. Enable the relevant exports when generating inputs for post-hoc analysis. Ablation notebooks currently write to the same `../4.Post Hoc/Best_df_user_MULTI.csv` path as the full model; assign distinct output paths before running ablations to preserve the full-model comparison.

## Reproducibility notes

- Seeds are applied to Python, NumPy, and PyTorch.
- The notebooks use CPU tensors directly and do not currently include explicit CUDA/MPS device handling.
- PyTorch deterministic-algorithm settings are not enabled, so exact results may vary across platforms or library versions.
- The training loader is shuffled, while validation and test loaders are not.
- Interpret accuracy together with macro F1 and MAE when evaluating class-imbalanced craving labels.
- One baseline record is removed by positional index before `user_order` is regenerated. Verify this step if the upstream table changes.
- The configuration table above describes `3.Modeling/Best_EPOCH_Multimodal.ipynb`. The notebooks in `5.Sliding Windows/` use hidden size 8; do not assume every experiment shares the primary model configuration.
- Windows are assigned to splits by their final input-row index, whose stored target is already shifted to the next observation. Validation/test windows can include earlier historical inputs. This evaluates future observations for known participants rather than generalization to unseen participants.
- The target is constructed with a within-participant `shift(-1)`. Window construction checks consecutive input days and, when a following row exists, its day gap. The final retained row does not receive that following-row gap check.
- Original data cleaning includes fixed row-index exclusions and participant exclusions. These steps assume the original row order and must be reviewed for a different dataset.
- The original executions of `HRV_Multimodal.ipynb` and `PF_Multimodal.ipynb` recorded a `KeyError` for `target_day ` (a trailing space) in a downstream prediction-summary cell. Review that column reference before a full rerun.

## Data availability and contact

Notebook outputs, execution counts, and execution metadata are removed from the public copies to avoid exposing participant-level tables and figures. Local research originals are preserved separately.

The data used in this study contain sensitive medical and human-participant information and **cannot be made publicly available**. All CSV files are excluded from this public code release, including raw clinical/EMA records, processed datasets, selected-feature tables, participant summaries, and participant-level prediction or performance tables.

Researchers interested in accessing the data should contact the corresponding author of the associated paper. Requests will be considered subject to applicable ethical approvals, participant consent, institutional policies, and any required data-use agreement. Contacting the authors does not guarantee access.

**Data-access contact:** Corresponding author of the associated paper. A direct contact email will be added with the final publication details.

Please do not include participant information or restricted datasets in public GitHub issues.

## Citation

This repository accompanies the accepted paper:

**Forecasting Next-Day Alcohol Craving in Individuals with Problematic Alcohol UseDrinkers w With Multimodal Data and a Tiny Recurrent Neural Network: Prospective Longitudinal Study**

The author list, journal, and DOI will be added when the publication details are available. Please cite the associated paper when using this code.

<!-- Replace this note with the verified citation and BibTeX entry. -->

## License

No software license is currently provided in this directory. Data-access permission is separate from software licensing and does not authorize redistribution of participant data.
