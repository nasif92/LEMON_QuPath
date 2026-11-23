# LEMON + QuPath: results summary

Nov 23, 2026 · @Nasif Hossain

## Question and answer

Frozen LEMON embeddings improve on QuPath alone in cross-validation on the 86 training slides (+4.3 points slide-mean accuracy with a two-scale 10 + 50 µm crop, p < 0.001), but not on the 18 untouched reference slides (+0.06 points, p = 0.93). The reference slides are the cleaner test, so no LEMON gain on held-out cases has been shown. Earlier 10 µm results on the reference slides: +0.4 points over all 108 QuPath features (p = 0.49), and LEMON alone about +2.3 over QuPath's own-nucleus features.

**Task.** Tumor vs Non Tumor classification of nuclei in ER, PR and Ki67 IHC breast cancer slides. Other labels (Ghost, Normal Epithelium, MIXED, Artifact, BadTissue) are dropped.

**Test data.** 18 reference slides, never used for training or tuning: 6 cases, each stained for ER, PR and Ki67. Test nuclei: Ki67 20,769, PR 10,016, ER 16,072.

**Training data.** One model per stain, trained on annotation phases 1 to 3 with the updated phase 2 and phase 3 annotations: Ki67 179,999 nuclei (28 slides), PR 120,900 (28), ER 167,045 (30). The first 10 µm reference run, reported below, used the earlier annotation set (Ki67 185,526, PR 123,231, ER 170,590).

**Model.** MLP with hidden layers 96-48-24-12-6, dropout 0.5, learning rate 1e-4, batch 10,000, class reweighting power 0.5, 200 epochs with a cosine schedule, final epoch kept. Three seeds (42, 43, 44).

**Inputs.** QuPath features (108 including aspect ratio; 44 neighbourhood, 64 own-nucleus). LEMON: frozen ViT-S/8, 384 dimensions per crop, each crop resized to 40 x 40. The first runs used a 10 µm crop centred on each nucleus (about 38 px at the slides' 0.263 µm/px). Later runs tried 15, 20, 30 and 50 µm crops and a two-scale input: the 10 µm and 50 µm embeddings concatenated (768 dimensions, 876 inputs with QuPath).

**Metric.** Slide-mean accuracy: the mean over the 6 reference slides of each slide's accuracy, averaged over seeds.

## Reference slides, 10 µm crop: results by stain

LEMON plus all QuPath features gains on Ki67 and PR and loses on ER; the three-stain average is 0.914 against 0.910. Slide-mean accuracy, mean ± SD over 3 seeds, on the 6 reference slides per stain.

| Configuration | Ki67 | PR | ER | Mean of 3 stains |
| --- | --- | --- | --- | --- |
| QuPath all (108) | 0.840 ± 0.002 | 0.931 ± 0.004 | **0.958 ± 0.001** | 0.910 |
| LEMON + QuPath all | **0.852 ± 0.001** | **0.940 ± 0.004** | 0.950 ± 0.001 | 0.914 |
| LEMON + QuPath all, PCA to 32 | 0.838 ± 0.011 | 0.922 ± 0.020 | 0.953 ± 0.004 | 0.904 |
| LEMON alone (384) | 0.822 ± 0.002 | 0.911 ± 0.002 | 0.882 ± 0.002 | 0.872 |
| QuPath neighbourhood (44) | 0.818 ± 0.008 | 0.904 ± 0.011 | 0.949 ± 0.004 | 0.890 |
| LEMON + QuPath neighbourhood | 0.846 ± 0.007 | 0.937 ± 0.008 | 0.951 ± 0.003 | 0.912 |
| QuPath own-nucleus (64) | 0.769 ± 0.003 | 0.883 ± 0.004 | 0.896 ± 0.002 | 0.849 |

The first six rows were planned before any result was seen. The own-nucleus row was added afterwards to test whether LEMON replaces those features, so treat it as exploratory.

The own-nucleus row comes from separate per-stain runs. A single run trained on all three stains together scored 0.764 (Ki67), 0.891 (PR) and 0.912 (ER); ER is 1.6 points higher than the per-stain run, more than the seed spread. This was not studied further.

## Paired comparisons over 18 slides

Only the neighbourhood comparison reaches p < 0.05, and the 18 slides are not independent. Each comparison pairs the same slide under two configurations, averaged over 3 seeds.

| A vs B | Mean difference (A - B) | A better on | Paired t-test |
| --- | --- | --- | --- |
| LEMON + all vs QuPath all | +0.44 points | 11 of 18 | t = 0.70, p = 0.49 |
| LEMON + neighbourhood vs QuPath neighbourhood | +2.16 points | 13 of 18 | t = 2.79, p = 0.012 |
| LEMON + neighbourhood vs QuPath all | +0.20 points | 8 of 18 | t = 0.33, p = 0.75 |
| LEMON alone vs QuPath own-nucleus (64) | +2.26 points | 11 of 18 | not recorded |

**Slides are grouped by case.** The 18 slides are 6 cases stained three ways. Averaging the neighbourhood difference per case gives +2.2, +1.3, +0.9, +1.4, +2.1 and +5.1 points: all six cases are positive. A t-test on those six numbers gives roughly t = 3.4, p = 0.02 (my arithmetic from the per-slide table, not a script output).

**LEMON alone vs own-nucleus features is uneven.** The median slide gains 1.4 points; four slides gain 7 to 12 points (Ki67 MIS, PR FHS, Ki67 RVS, Ki67 FHS). By stain: Ki67 +5.3 points (5 of 6 slides better), PR +2.8 (4 of 6), ER -1.4 (2 of 6).

**One slide stands out.** Ki67 RDS24-012921 is worse with LEMON in the all-feature and neighbourhood comparisons (-6.6 and -4.5 points) and one of the two hardest slides (accuracy about 0.72). Nothing has been checked on why.

## Cross-validation on the training slides, 10 µm crop

To test LEMON before touching the reference slides, I ran 5-fold cross-validation grouped by slide on the 86 training slides (Ki67 28, PR 28, ER 30), with the same folds for every configuration. Fusion beats QuPath all by +1.8 points (p = 0.014); LEMON alone beats own-nucleus features by +2.4 (p < 0.001). Slide-mean accuracy, seed 42, updated annotations.

| Configuration | Ki67 | PR | ER | Mean of 3 stains |
| --- | --- | --- | --- | --- |
| QuPath all (108) | 0.8465 | 0.8469 | 0.9056 | 0.8663 |
| LEMON + QuPath all | 0.8711 | 0.8795 | 0.9047 | 0.8851 |
| LEMON + QuPath all, PCA to 32 | 0.8373 | 0.8405 | 0.9015 | 0.8598 |
| LEMON alone | 0.8365 | 0.8671 | 0.8977 | 0.8671 |
| QuPath neighbourhood (44) | 0.8334 | 0.8118 | 0.8274 | 0.8242 |
| LEMON + QuPath neighbourhood | 0.8700 | 0.8793 | 0.9030 | 0.8841 |
| QuPath own-nucleus (64) | 0.8004 | 0.8355 | 0.8931 | 0.8430 |

Paired over the 86 held-out slides (A - B), 10 µm:

| A vs B | Mean difference | A better on | p |
| --- | --- | --- | --- |
| LEMON + all vs QuPath all | +1.83 points | 52 of 86 | 0.014 |
| LEMON + neighbourhood vs QuPath neighbourhood | +6.03 points | 57 of 86 | < 0.001 |
| LEMON + neighbourhood vs QuPath all | +1.73 points | 44 of 86 | 0.033 |
| LEMON alone vs QuPath own-nucleus | +2.36 points | 64 of 86 | < 0.001 |

The gain is concentrated in Ki67 (+2.5) and PR (+3.3); ER is a tie (-0.1). A first cross-validation on the earlier phase 2 annotations (85 slides) gave the same pattern: fusion +2.1 points over QuPath all, p = 0.002.

## Crop size and two-scale input

Larger crops help LEMON alone, fusion plateaus from 15 µm, and concatenating the 10 µm and 50 µm embeddings is best for every stain. Slide-mean accuracy on the training slides, seed 42, 5-fold cross-validation, same folds throughout.

| Crop | LEMON alone, mean | LEMON + QuPath all: Ki67 | PR | ER | Mean |
| --- | --- | --- | --- | --- | --- |
| 10 µm | 0.8671 | 0.8711 | 0.8795 | 0.9047 | 0.8851 |
| 15 µm | 0.8765 | 0.8806 | 0.8820 | 0.9091 | 0.8906 |
| 20 µm | 0.8813 | 0.8858 | 0.8826 | 0.9076 | 0.8920 |
| 30 µm | 0.8844 | 0.8930 | 0.8793 | 0.9040 | 0.8921 |
| 50 µm | 0.8832 | 0.8977 | 0.8709 | 0.9081 | 0.8922 |
| 10 + 50 µm (768 dimensions) | 0.8972 | 0.9055 | 0.8923 | 0.9154 | 0.9044 |

Single crops trade stains against each other: Ki67 keeps improving up to 50 µm (0.871 to 0.898) while PR peaks at 20 µm and falls to 0.871 at 50 µm. The two-scale input beats the best single size on all three stains. The sweep is one seed, and I chose the two-scale pair after seeing it, so treat it as exploratory.

## Cross-validation with the two-scale input (3 seeds)

With the 10 + 50 µm input, fusion beats QuPath all by +4.3 points (p < 0.001) and LEMON alone (0.898) beats all 108 QuPath features (0.858). Slide-mean accuracy on 86 training slides, mean over seeds 42, 43 and 44.

| Configuration | Ki67 | PR | ER | Mean of 3 stains |
| --- | --- | --- | --- | --- |
| QuPath all (108) | 0.8451 | 0.8286 | 0.9007 | 0.8581 |
| LEMON + QuPath all | 0.9048 | 0.8912 | 0.9102 | 0.9020 |
| LEMON + QuPath all, PCA to 32 | 0.8661 | 0.8566 | 0.9082 | 0.8770 |
| LEMON alone | 0.8928 | 0.8901 | 0.9120 | 0.8983 |
| QuPath neighbourhood (44) | 0.8345 | 0.8105 | 0.8317 | 0.8256 |
| LEMON + QuPath neighbourhood | 0.9062 | 0.8889 | 0.9116 | 0.9022 |
| QuPath own-nucleus (64) | 0.8012 | 0.8383 | 0.8942 | 0.8446 |

| A vs B | Mean difference | A better on | p |
| --- | --- | --- | --- |
| LEMON + all vs QuPath all | +4.31 points | 65 of 86 | < 0.001 |
| LEMON + neighbourhood vs QuPath neighbourhood | +7.67 points | 74 of 86 | < 0.001 |
| LEMON + neighbourhood vs QuPath all | +4.33 points | 66 of 86 | < 0.001 |
| LEMON alone vs QuPath own-nucleus | +5.29 points | 70 of 86 | < 0.001 |

By stain, LEMON + all vs QuPath all is Ki67 +6.0, PR +6.3 and ER +1.0. LEMON + neighbourhood (0.9022) matches LEMON + all (0.9020) with 64 fewer features. QuPath all on PR is unstable across seeds (0.847, 0.794, 0.846), which inflates the PR gap somewhat.

By annotation phase of the held-out slide (accuracy, mean over slides):

| Configuration | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| QuPath all | 0.9581 | 0.8499 | 0.7827 |
| LEMON alone | 0.9630 | 0.8780 | 0.8634 |
| LEMON + QuPath all | 0.9728 | 0.8859 | 0.8574 |
| LEMON + QuPath neighbourhood | 0.9725 | 0.8838 | 0.8604 |

Phase 1 slides are near ceiling for every configuration; the gain is largest on phase 3 slides (+7.5 points). The p-values treat slides as independent, and the three seeds are averaged rather than counted as extra samples.

## Reference slides with the two-scale input

On the 18 reference slides, LEMON + QuPath all and QuPath all tie: 0.9110 against 0.9104, fusion better on 10 of 18 slides, paired t-test p = 0.93. The cross-validation gain did not carry over. Slide-mean accuracy, mean ± SD over 3 seeds; training on the updated annotations, nothing tuned on these slides.

| Configuration | Ki67 | PR | ER | Mean of 3 stains |
| --- | --- | --- | --- | --- |
| QuPath all (108) | 0.8426 ± 0.0018 | 0.9326 ± 0.0022 | 0.9561 ± 0.0008 | 0.9104 |
| LEMON (10 µm) + QuPath all | 0.8509 ± 0.0015 | 0.9365 ± 0.0014 | 0.9479 ± 0.0004 | 0.9118 |
| LEMON (10 + 50 µm) + QuPath all | 0.8429 ± 0.0063 | 0.9456 ± 0.0029 | 0.9446 ± 0.0024 | 0.9110 |
| Difference, 10 µm minus QuPath | +0.83 | +0.39 | -0.82 | +0.13 |
| Difference, 10 + 50 µm minus QuPath | +0.03 | +1.30 | -1.15 | +0.06 |

Per slide, averaged over seeds (A = LEMON + QuPath all, B = QuPath all):

| Slide | A | B | A - B |
| --- | --- | --- | --- |
| Ki67 RDS24-012921 | 0.6586 | 0.7218 | -6.3 |
| ER RDS24-012921 | 0.9078 | 0.9417 | -3.4 |
| ER RVS24-009627 | 0.9166 | 0.9448 | -2.8 |
| Ki67 FHS24-019049 | 0.7015 | 0.7233 | -2.2 |
| ER FHS24-019049 | 0.8956 | 0.9172 | -2.2 |
| PR FHS24-019049 | 0.9583 | 0.9769 | -1.9 |
| PR RVS24-009627 | 0.9632 | 0.9697 | -0.7 |
| Ki67 CIS24-000938 | 0.9211 | 0.9226 | -0.2 |
| ER MIS24-007597 | 0.9882 | 0.9860 | +0.2 |
| PR PLS24-009859 | 0.9781 | 0.9744 | +0.4 |
| Ki67 MIS24-007597 | 0.9338 | 0.9286 | +0.5 |
| ER PLS24-009859 | 0.9782 | 0.9730 | +0.5 |
| ER CIS24-000938 | 0.9812 | 0.9740 | +0.7 |
| PR RDS24-012921 | 0.8915 | 0.8766 | +1.5 |
| Ki67 RVS24-009627 | 0.9345 | 0.9060 | +2.9 |
| PR CIS24-000938 | 0.9442 | 0.9150 | +2.9 |
| Ki67 PLS24-009859 | 0.9080 | 0.8533 | +5.5 |
| PR MIS24-007597 | 0.9382 | 0.8828 | +5.5 |

Fusion loses most on slides from cases RDS24-012921 and FHS24-019049, which are also among the lowest-accuracy slides for both models. It gains on a different set (Ki67 PLS24-009859, PR MIS24-007597). Eighteen slides from six cases cannot detect a one or two point difference. Re-run at 10 µm on the updated annotations (3 seeds), fusion also ties QuPath all: +0.13 points, better on 10 of 18 slides, p = 0.68 (ER -0.8, Ki67 +0.8, PR +0.4). The two-scale input did not improve the reference result over 10 µm (-0.07 points, p = 0.88). The earlier 10 µm run on the previous annotations gave +1.2 (Ki67), +0.9 (PR) and -0.8 (ER).

Training accuracy at the end of training is 0.984 for fusion and 0.960 for QuPath all; the extra fit does not carry to the reference slides.

## Earlier QuPath-only experiments (Ki67)

Neighbourhood features carry most of the signal in QuPath alone, and no mixture-of-experts variant beat a single MLP. Ki67 slide-mean accuracy on the same 6 reference slides, 3 seeds, trained on 28 slides with no holdout.

| Feature group | Features | Slide-mean accuracy |
| --- | --- | --- |
| Shape | 7 | 0.755 |
| Intensity | 57 | 0.745 |
| Neighbourhood (context only) | 44 | 0.814 |
| All, with aspect ratio | 108 | 0.837 |

- The 108-feature baseline in this project (0.840) matches the earlier 0.837, so the standalone code reproduces the original setup.
- Per-slide normalisation of DAB made no difference (0.827 and 0.826 against 0.832) and was dropped.
- Mixture-of-experts: no gate type or grouping beat the plain MLP, and a uniform gate matched the learned gate.

Not summarised here: the sub-feature MLPs (Haralick, neighbourhood texture, intensity and shape) and the Cellpose-DINO tumor-only fine-tuning runs. Their numbers are in your earlier logs, not in this document.

## Conclusions and limits

Cross-validation supports a LEMON gain and the reference slides do not; both results should be reported together.

1. In slide-grouped 5-fold cross-validation on 86 training slides, LEMON + QuPath with the 10 + 50 µm input beats QuPath all by +4.3 points (65 of 86 slides, p < 0.001; Ki67 +6.0, PR +6.3, ER +1.0). The gain is smaller at 10 µm alone (+1.8 to +2.1, p = 0.002 to 0.014).
2. On the 18 reference slides at the same setting, the two tie: 0.9110 against 0.9104, fusion better on 10 of 18 slides, p = 0.93 (PR +1.3, Ki67 0.0, ER -1.2).
3. LEMON alone beats QuPath's own-nucleus features in cross-validation (+5.3 points, p < 0.001) and beats all 108 QuPath features (0.898 against 0.858). On the reference slides at 10 µm it was +2.3 over own-nucleus features.
4. Crop size matters for LEMON alone and little for fusion: LEMON alone rises from 0.867 (10 µm) to 0.884 (30 µm); fusion plateaus near 0.892 from 15 µm up. Combining 10 and 50 µm gives the best cross-validation result for every stain (fusion 0.904).
5. PCA to 32 components lowers accuracy in every comparison; keep the full embedding.

**Limits.**

- Six cases. The 18 reference slides are not independent, single slides swing by 5 to 9 points, and a difference of one or two points cannot be detected.
- Crop sizes and the two-scale input were chosen by cross-validation on the training slides, so those numbers are somewhat optimistic. The reference slides were not used for any choice.
- The crop-size sweep is one seed (42) and one fold split; only the final 10 + 50 µm cross-validation used three seeds. No paired tests were run between crop sizes.
- LEMON was trained on H&E, not IHC, and is used frozen. No other frozen encoder was tried, so the result says a pretrained histology embedding helps in cross-validation, not that LEMON is the best choice.
- Nuclei scored against the labelled-binary classification labels (2.76 million nuclei) were not embedded, so that test set has not been scored.
- ER is near its ceiling (about 0.90 in cross-validation, 0.95 on the reference slides), so there was little room to gain.
- The own-nucleus comparison was added after seeing results, and the combined-stain training result for ER differs from per-stain training.

**Open items.**

- Find out why the cross-validation gain does not carry to the reference slides. One untested idea is a stain or scan shift between cohorts that affects embeddings more than QuPath's measurements.
- Check the two slides that are worst for both models: Ki67 RDS24-012921 and Ki67 FHS24-019049 (fusion loses 6.3 and 2.2 points on them).
- If a shift is found, try stain normalisation before extraction and judge it by cross-validation.
- Compare against another frozen encoder to see how much of the gain is specific to LEMON.
- Decide whether to score a subsample of the labelled-binary nuclei.
- Decide whether to report the per-stain or combined-stain own-nucleus baseline.
