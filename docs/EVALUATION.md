# Evaluation: how the model is tested, and how far the tests can be trusted

[METHODOLOGY.md](METHODOLOGY.md) describes what was done. This document describes how each claim is checked, what
each check can and cannot show, and what could still be wrong. The guiding question for every number is the same:
compared with what, and how sure are we?

## Who decides what

| Data | Rows | Decides | Never used for |
|---|---|---|---|
| training | 14,448 | weights and the scaler's statistics | any choice between models |
| validation | 3,096 | epochs, width, activation, hyperparameters and interval width | the reported test score |
| test | 3,096 | the reported performance, once | anything else |
| training and validation combined | 17,544 | cross-validation estimates, with the scaler refitted in every fold | selection |

Keeping these roles apart matters because a score that has influenced a decision is no longer an unbiased estimate of
future performance. Choosing the best of many configurations on the validation set biases that set's score
downwards, which is why the validation score is never reported as the model's performance (Cawley & Talbot, 2010).

## Metrics

Training minimises mean squared error, which is smooth and penalises large errors heavily. Results are reported
mainly as mean absolute error in dollars, because "a typical miss of $X" is easier to reason about than a squared
quantity, with RMSE and R² alongside. RMSE is always at least as large as MAE, and the gap between them indicates how
much of the error comes from a minority of large misses.

## How uncertain is the headline number?

The test set is one sample of 3,096 block groups, and a different but equally plausible sample would give a different
MAE. The percentile bootstrap estimates that variation by resampling the test rows with replacement 2,000 times and
reading the 2.5th and 97.5th percentiles of the resampled metric (Efron & Tibshirani, 1993). This interval describes
sampling variation of the test set only. It does not cover variation from re-training, which the seed experiments and
cross-validation address separately.

## Is one model really better than another?

Comparing two separate intervals is a weak test, because the two models' errors on the same rows are correlated. Every
model is therefore scored on the same resamples, which yields a paired interval for the difference in MAE. When that
interval excludes zero, the difference is unlikely to be an artefact of this particular test sample. The reported
two-sided bootstrap p-value is a description of the same resampling distribution, not a guarantee, and with eight
hypotheses and several comparisons the reader should look for consistent patterns rather than single small p-values.

## Does the model work beyond a random split?

California's block groups are spatially autocorrelated: nearby rows share income levels, housing stock and prices. A
random split places neighbours on both sides of the train/test line, so part of the test score measures interpolation
between known neighbours. Spatial block cross-validation holds out whole regions at a time, and the difference between
its error and the error of random folds measures how optimistic a random split is for new places (Roberts et al.,
2017). Both use the training and validation rows only, refit the scaler in every fold, and train for the number of
epochs that was fixed before the test evaluation (Kohavi, 1995).

## Are the stated uncertainties honest?

Conformal intervals promise a minimum coverage on average (Lei et al., 2018). The promise is checked rather than
trusted: the notebook reports the coverage achieved on the test set at six levels, from 50% to 95%. Two qualifications
apply. The validation set that calibrates the intervals also selected the model, so the calibration rows are not
perfectly exchangeable with new rows. And coverage is marginal, so it can be lower for identifiable groups. Coverage is
reported separately at the price ceiling and by city for this reason (Angelopoulos & Bates, 2023).

## Who bears the errors?

Average error can hide groups for whom the model works poorly. Slice tests break the test error down by nearest major
city, income band, the price ceiling and the presence of extreme inputs, each with a bootstrap interval. The data hold
no demographic attributes, so this is not a fairness audit in the legal sense. It still shows where a user of the
model should be most careful.

## Watching the inputs

A model can only be trusted for inputs that resemble its training data. The batch-scoring tools compare new data with
the training distribution in two ways. Per feature, the population stability index and the two-sample
Kolmogorov–Smirnov distance measure how far each distribution has moved (Siddiqi, 2006; Massey, 1951). Per row, the
Mahalanobis distance in the model's input space flags points beyond the training data's 99th percentile (Mahalanobis,
1936). Simple per-feature tests of this kind are a strong baseline for detecting dataset shift (Rabanser et al., 2019).

## Threats to validity

The framework of Shadish et al. (2002) organises what could still be wrong.

- **Internal validity: are observed differences caused by the factor being varied?** Training noise could masquerade
  as an effect, so every configuration runs on five seeds. Leakage could inflate scores, so the scaler sees training
  rows only, and an audit quantifies what leakage would have changed. Hidden non-determinism could make comparisons
  unrepeatable, so deterministic kernels are used and the baseline is re-trained to confirm an identical curve.
- **Statistical conclusion validity: are the inferences sound?** The bootstrap treats test rows as independent, but
  spatial dependence means neighbouring rows carry overlapping information, so the intervals are probably too narrow;
  a spatial block bootstrap would widen them and is a natural next step. Five folds give only a rough estimate of
  fold-to-fold variation. Several comparisons are made without a multiplicity correction.
- **Construct validity: do the measurements mean what we say?** The target is a block-group median, top-coded at
  $500,000, so errors at the top of the market are understated (Tobin, 1958). MSE and MAE weight all dollars equally,
  although a $50,000 miss matters more for a $150,000 home than for a $450,000 one.
- **External validity: how far do the results travel?** The data describe California in 1990, at the level of block
  groups. Nothing here supports claims about today's prices, other markets or individual homes (Robinson, 1950).

## What the software tests guarantee

Machine-learning systems fail through their data and their models as often as through their code, so the test suite
checks all three (Breck et al., 2017).

| Test file | Guarantee |
|---|---|
| `test_model.py` | the NumPy forward pass matches a hand computation for every activation and reproduces the Keras predictions |
| `test_artifacts.py` | missing, damaged or altered artifacts are rejected with a clear message; checksums are verified |
| `test_storage.py` | local and S3 storage behave identically; records are append-only; settings precedence is correct |
| `test_evaluation.py` | bootstrap intervals contain the estimate and shrink with more data; paired comparisons detect a real gap |
| `test_uncertainty.py` | conformal intervals reach their promised coverage on heavy-tailed synthetic data |
| `test_explain.py` | Shapley values add up exactly, are exact for linear models and give unused features zero |
| `test_monitoring.py` | invalid rows are caught; PSI and KS separate stable from shifted data |
| `test_service.py`, `test_cli.py`, `test_api.py` | every interface returns ordered intervals, sensible errors and explanations that add up |
| `test_report.py` | the report is generated, flags synthetic fixtures and passes its own freshness check |
| `test_app.py` | every dashboard page renders without an exception |

## How to verify the published numbers

```bash
python -m housing_app validate        # checksums in artifacts/manifest.json and the NumPy-against-Keras self-check
python -m housing_app report --check  # docs/RESULTS.md was generated from exactly the committed artifacts
python -m housing_app report          # recompute every number in docs/RESULTS.md
```

## References

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591. https://doi.org/10.1561/2200000101

Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML test score: A rubric for ML production readiness and technical debt reduction. In *2017 IEEE International Conference on Big Data* (pp. 1123–1132). IEEE. https://doi.org/10.1109/BigData.2017.8258038

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In *Proceedings of the 14th International Joint Conference on Artificial Intelligence* (Vol. 2, pp. 1137–1143). Morgan Kaufmann.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association, 113*(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116

Mahalanobis, P. C. (1936). On the generalised distance in statistics. *Proceedings of the National Institute of Sciences of India, 2*(1), 49–55.

Massey, F. J., Jr. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association, 46*(253), 68–78. https://doi.org/10.1080/01621459.1951.10500769

Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. In *Advances in Neural Information Processing Systems 32*. Curran Associates. https://doi.org/10.48550/arXiv.1810.11953

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review, 15*(3), 351–357. https://doi.org/10.2307/2087176

Shadish, W. R., Cook, T. D., & Campbell, D. T. (2002). *Experimental and quasi-experimental designs for generalized causal inference*. Houghton Mifflin.

Siddiqi, N. (2006). *Credit risk scorecards: Developing and implementing intelligent credit scoring*. Wiley.

Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica, 26*(1), 24–36. https://doi.org/10.2307/1907382
