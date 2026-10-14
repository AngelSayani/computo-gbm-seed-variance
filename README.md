# Seed variance in gradient-boosted tree pipelines for industrial
sensor-based fault detection
Angel Sayani
2026-10-14

### Citation

Submitted to Computo.

### Badges

[![build and
publish](https://github.com/AngelSayani/computo-gbm-seed-variance/actions/workflows/build.yml/badge.svg)](https://github.com/AngelSayani/computo-gbm-seed-variance/actions/workflows/build.yml)
[![reviews](https://img.shields.io/badge/review-report-blue)](https://github.com/AngelSayani/computo-gbm-seed-variance/issues?q=is:open is:issue label:review)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/AngelSayani/computo-gbm-seed-variance)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/AngelSayani/computo-gbm-seed-variance)
[![DOI:10.5072/computo.0000](https://img.shields.io/badge/DOI-10.5072%2Fcomputo.0000-034E79.svg)](https://doi.org/10.5072/computo.0000)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Authors’ affiliations

- [Angel Sayani](https://www.linkedin.com/in/angelsayani) (IntellChromatics LLC)

### Abstract

Industrial fault-detection systems are increasingly built on
gradient-boosted tree (GBT) classifiers trained on sensor data, where
the positive class (a failure event) is rare. Reporting a single
train/test split and a single model fit is common practice in this
literature, yet both steps involve pseudo-random choices. We decompose
the seed-induced variance of two widely used scikit-learn GBT
implementations on a real, publicly available industrial fault-detection
dataset (AI4I 2020, 10 000 operating records, 3.39% failure rate) into
the share coming from the train/test split and the share coming from the
model’s own internal randomness, under two class-imbalance-handling
settings. We find that (i) a default-configured
`HistGradientBoostingClassifier` is, on this dataset, entirely
deterministic given a fixed split and therefore contributes zero
model-level variance, while the same classifier with
`early_stopping=True` and the classic `GradientBoostingClassifier` with
row subsampling both show real, non-negligible model-seed variance; (ii)
across every configuration we test, a single unlucky or lucky train/test
split alone can move the test PR-AUC by 8-17 percentage points relative
to the 60-seed mean, which is larger than the differences researchers
often report between competing methods; and (iii) whether a GBT
implementation’s declared `random_state` actually matters depends on
implementation details (data size relative to the library’s
auto-early-stopping threshold, subsampling settings) that are easy to
overlook. We give a reproducible, seed-decomposition protocol and code
that practitioners and reviewers can reuse on their own fault-detection
pipelines to check whether a reported result is a property of the method
or a property of the seed.
