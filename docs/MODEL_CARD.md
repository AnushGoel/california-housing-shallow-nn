# Model card: California housing shallow network

This card follows the structure proposed by Mitchell et al. (2019).

## Model details

- **Developer and version.** Anush Goel, 2026; version 2.1.0, released as an open research project under the MIT License.
- **Type.** A feed-forward neural network with one hidden layer and a linear output unit, trained with Adam on mean
  squared error (Kingma & Ba, 2014). The width and the activation were chosen by controlled, seed-replicated
  experiments described in [METHODOLOGY.md](METHODOLOGY.md). A tuned variant, the "champion", is also provided.
- **Inputs.** Eight block-group features: median income, median house age, average rooms, bedrooms and occupants per
  household, population, latitude and longitude.
- **Output.** The median house value of the block group in 1990 dollars, with a split-conformal prediction interval
  (Lei et al., 2018) and, on request, an exact Shapley breakdown of the prediction.

## Intended use

- Learning how capacity, activation and evaluation choices shape a small neural network.
- Exploring neighbourhood-level housing patterns in California in 1990.
- A worked example of honest evaluation and of serving a model with intervals, explanations and drift checks.

## Out-of-scope uses

- Valuing any individual property, in 1990 or today.
- Lending, insurance, taxation or any other decision about people. Housing values in 1990 reflect, among other things,
  decades of discriminatory lending and zoning (Rothstein, 2017), and a model that learns those values learns their
  history. Nothing here has been audited for such effects.
- Housing markets outside California, or any year other than 1990.

## Factors

Performance varies with geography (coastal metropolitan areas against inland regions, and regions absent from the
training data), with price level (values are top-coded at $500,000) and with unusual block groups whose
per-household averages are extreme. [RESULTS.md](RESULTS.md) reports errors and interval coverage for each of these.

## Metrics

<!-- results:start -->
|  | result |
|---|---|
| Typical error (test MAE) | $34,929 (95% CI $33,614 to $36,319) |
| Variance explained (R²) | 0.801 |
| 90% prediction interval | ±$82,868, covered 91.0% of test block groups |
| Cost of predicting unseen regions | +20.7% MAE (spatial against random folds) |
| Artifacts fingerprint | `f0d7955fc3a6` |

Full evidence, figures and conclusions: [RESULTS.md](RESULTS.md)
<!-- results:end -->

Every metric carries a 95% bootstrap interval. Interval coverage is measured on the test set, and spatial
cross-validation estimates the error for regions the model has not seen.

## Evaluation data

The test set is a seeded random 15% of the block groups (3,096 rows), scored once after every modelling decision.
Spatial cross-validation on the remaining 85% holds out whole regions to estimate performance on unseen places.

## Training data

70% of the block groups (14,448 rows) fit the weights, and 15% (3,096 rows) select the configuration and calibrate the
intervals. Features are standardised with training statistics; the tuned champion may also log-transform four
heavy-tailed features. [DATASHEET.md](DATASHEET.md) describes the data.

## Ethical considerations

The data are aggregates with no personal information, but they are not neutral: they encode the housing market of
1990, including its inequities. Errors are not shared evenly, and the slice tests in [RESULTS.md](RESULTS.md) show
which places receive the largest ones.

## Caveats and recommendations

- Read predictions as 1990 neighbourhood medians, not as prices of homes.
- Use the interval rather than the point estimate, and treat an out-of-distribution flag as a warning that the model
  is extrapolating.
- Expect understated errors for neighbourhoods at the $500,000 ceiling.
- Where accuracy matters most, gradient-boosted trees are a strong alternative on tabular data of this kind
  (Grinsztajn et al., 2022).

## References

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization* (arXiv:1412.6980). arXiv. https://doi.org/10.48550/arXiv.1412.6980

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association, 113*(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596

Rothstein, R. (2017). *The color of law: A forgotten history of how our government segregated America*. Liveright.
