# Synthetic validation report

Local (DSP/VAD/pyannote) pipeline scored against constructed ground truth.
Corpus: 52 clips derived from the three production calls (audio untracked; construction in `analysis/synthetic_validation.py`).

## noise_severity — accuracy 84.62% (n=52)

| gt \ pred | high | low | medium | none |
|---|---|---|---|---|
| **high** | 8 | 1 | 1 | 2 |
| **low** | 0 | 6 | 0 | 2 |
| **medium** | 0 | 1 | 6 | 1 |
| **none** | 0 | 0 | 0 | 24 |

## noise_present — accuracy 90.38% (n=52)

| gt \ pred | False | True |
|---|---|---|
| **False** | 24 | 0 |
| **True** | 5 | 23 |

## quality — accuracy 90.91% (n=55)

| gt \ pred | clear | severely_impaired | slightly_impaired |
|---|---|---|---|
| **clear** | 31 | 0 | 0 |
| **severely_impaired** | 0 | 5 | 1 |
| **slightly_impaired** | 4 | 0 | 14 |

## long_silence — accuracy 100.00% (n=55)

| gt \ pred | False | True |
|---|---|---|
| **False** | 50 | 0 |
| **True** | 0 | 5 |

## static — accuracy 96.15% (n=52)

| gt \ pred | False | True |
|---|---|---|
| **False** | 46 | 2 |
| **True** | 0 | 4 |

## overlap — accuracy 80.00% (n=5)

| gt \ pred | False | True |
|---|---|---|
| **False** | 2 | 0 |
| **True** | 1 | 2 |

## fused noise on babble subset (local + Gemini)

- `c1_control`: gt **none** -> local none, llm low (road noise), fused **none** [llm low, uncorroborated -> none]
- `c1_babble_snr30`: gt **low** -> local none, llm medium (crowd chatter), fused **medium** [llm >= medium]
- `c1_babble_snr20`: gt **medium** -> local low, llm medium (radio and background chatter), fused **medium** [llm >= medium]
- `c1_babble_snr10`: gt **high** -> local low, llm high (overlapping speech and radio), fused **high** [llm >= medium]
- `c1_babble_snr5`: gt **high** -> local medium, llm high (muffled speech/chatter), fused **high** [llm >= medium]
- `c2_babble_snr30`: gt **low** -> local none, llm medium (TV or radio broadcast), fused **medium** [llm >= medium]
- `c2_babble_snr20`: gt **medium** -> local none, llm medium (TV or radio broadcast), fused **medium** [llm >= medium]
- `c2_babble_snr10`: gt **high** -> local none, llm medium (TV or radio broadcast), fused **medium** [llm >= medium]
- `c2_babble_snr5`: gt **high** -> local none, llm high (TV or radio broadcast), fused **medium** [llm high capped to medium (clean pauses)]

Speech-like noise, measured: presence 44% local-only vs **100% fused**; severity 11% local-only vs **56% fused** exact (100% within one level).
