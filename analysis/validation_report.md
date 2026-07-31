# Synthetic validation report

Local (DSP/VAD/pyannote) pipeline scored against constructed ground truth.
Corpus: 52 clips derived from the three production calls (audio untracked; construction in `analysis/synthetic_validation.py`).

## noise_severity — accuracy 84.62%, macro F1 0.814 (n=52)

| gt \ pred | high | low | medium | none |
|---|---|---|---|---|
| **high** | 8 | 1 | 1 | 2 |
| **low** | 0 | 6 | 0 | 2 |
| **medium** | 0 | 1 | 6 | 1 |
| **none** | 0 | 0 | 0 | 24 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| high | 1.00 | 0.67 | 0.80 | 12 |
| low | 0.75 | 0.75 | 0.75 | 8 |
| medium | 0.86 | 0.75 | 0.80 | 8 |
| none | 0.83 | 1.00 | 0.91 | 24 |

## noise_present — accuracy 90.38%, macro F1 0.904 (n=52)

| gt \ pred | False | True |
|---|---|---|
| **False** | 24 | 0 |
| **True** | 5 | 23 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| False | 0.83 | 1.00 | 0.91 | 24 |
| True | 1.00 | 0.82 | 0.90 | 28 |

## quality — accuracy 90.91%, macro F1 0.899 (n=55)

| gt \ pred | clear | severely_impaired | slightly_impaired |
|---|---|---|---|
| **clear** | 31 | 0 | 0 |
| **severely_impaired** | 0 | 5 | 1 |
| **slightly_impaired** | 4 | 0 | 14 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| clear | 0.89 | 1.00 | 0.94 | 31 |
| severely_impaired | 1.00 | 0.83 | 0.91 | 6 |
| slightly_impaired | 0.93 | 0.78 | 0.85 | 18 |

## long_silence — accuracy 100.00%, macro F1 1.000 (n=55)

| gt \ pred | False | True |
|---|---|---|
| **False** | 50 | 0 |
| **True** | 0 | 5 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| False | 1.00 | 1.00 | 1.00 | 50 |
| True | 1.00 | 1.00 | 1.00 | 5 |

## static — accuracy 96.15%, macro F1 0.889 (n=52)

| gt \ pred | False | True |
|---|---|---|
| **False** | 46 | 2 |
| **True** | 0 | 4 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| False | 1.00 | 0.96 | 0.98 | 48 |
| True | 0.67 | 1.00 | 0.80 | 4 |

## overlap — accuracy 80.00%, macro F1 0.800 (n=5)

| gt \ pred | False | True |
|---|---|---|
| **False** | 2 | 0 |
| **True** | 1 | 2 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| False | 0.67 | 1.00 | 0.80 | 2 |
| True | 1.00 | 0.67 | 0.80 | 3 |
