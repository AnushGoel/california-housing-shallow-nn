# Datasheet: California Housing, 1990 census block groups

This datasheet answers the questions proposed by Gebru et al. (2021). The project did not collect the data. The
answers come from the original publication (Pace & Barry, 1997), the scikit-learn documentation (scikit-learn
developers, n.d.) and this project's own exploration, which is in Section 1 of the notebook.

## Motivation

Pace and Barry (1997) assembled the data to demonstrate fast estimation of spatial autoregressive models on a large,
real dataset, and it has since become a standard regression benchmark distributed with scikit-learn. This project uses
it because it is public, well documented and large enough for careful testing, and because its known quirks (spatial
dependence, top-coding, heavy tails) make it a demanding test of honest evaluation.

## Composition

Each of the 20,640 rows is one census block group from the 1990 U.S. census. A block group is the smallest
geographical unit for which the Census Bureau published sample data, and typically holds 600 to 3,000 people. There
are no missing values.

| Column | Meaning | Unit |
|---|---|---|
| MedInc | median household income | tens of thousands of dollars |
| HouseAge | median age of the houses | years |
| AveRooms | average number of rooms per household | rooms |
| AveBedrms | average number of bedrooms per household | bedrooms |
| Population | residents of the block group | people |
| AveOccup | average number of people per household | people |
| Latitude, Longitude | location of the block group | degrees |
| MedHouseVal (target) | median house value | $100,000s |

The rows are aggregates, so they contain no information about individual people or homes. The per-household columns
are averages derived from block-group totals.

## Known quirks

| Quirk | Consequence | How the project handles it |
|---|---|---|
| target top-coded at $500,000 (about 5% of rows) | errors for expensive areas are understated (Tobin, 1958) | kept; errors and interval coverage reported separately |
| HouseAge capped at 52 and MedInc at 15 | the oldest housing and the highest incomes are compressed | kept; noted when interpreting partial dependence |
| extreme per-household averages, e.g. resort areas with few households | standardised values far from the mean | kept; a log transform is tested in the tuning study |
| spatial autocorrelation | random splits are optimistic for new regions (Roberts et al., 2017) | spatial cross-validation alongside random folds |
| aggregation to block groups | group patterns do not describe individuals (Robinson, 1950) | stated in every results document |

## Collection

The values come from the 1990 U.S. census. The version used here was obtained by scikit-learn from the StatLib
repository and is downloaded with `sklearn.datasets.fetch_california_housing` (scikit-learn developers, n.d.).

## Preprocessing in this project

No rows are removed. The rows are split 70/15/15 with a fixed seed, and each feature is standardised with training
statistics; the tuned champion can also apply a log transform to four heavy-tailed features. A compressed copy with the
split labels is stored as `artifacts/dataset.csv.gz` for the dashboard.

## Uses

The dataset suits regression benchmarks, teaching and research on evaluation methods. It is unsuitable for valuing
individual properties and outdated as a description of today's prices.

## Distribution and maintenance

The data are a static historical record, distributed with scikit-learn, and are not updated.

## References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters, 33*(3), 291–297. https://doi.org/10.1016/S0167-7152(96)00140-X

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review, 15*(3), 351–357. https://doi.org/10.2307/2087176

scikit-learn developers. (n.d.). *California Housing dataset*. In *scikit-learn user guide*. Retrieved October 5, 2026, from https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica, 26*(1), 24–36. https://doi.org/10.2307/1907382
