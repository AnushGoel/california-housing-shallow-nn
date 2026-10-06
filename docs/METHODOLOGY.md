# Methodology: what was done, and why

This document explains the design of the study behind the model: the questions it asks, the decision taken at each
step, and the reasoning and sources behind each one. The numbers live in [RESULTS.md](RESULTS.md), which is generated
from the trained artifacts, so nothing written here depends on a particular run.

## 1. Questions

The project asks two questions about one-hidden-layer neural networks on tabular data.

1. **How do capacity and activation shape learning?** How does the number of hidden units change underfitting,
   overfitting and validation error, and how do ReLU, tanh and sigmoid differ in convergence speed, stability and
   final accuracy?
2. **How far can the resulting model be trusted?** How large is its error on unseen block groups, how uncertain is
   that number, how does the model compare with simpler and stronger alternatives, how does it cope with regions it
   has never seen, and are its stated uncertainties honest?

The first question concerns learning dynamics and is answered with controlled experiments. The second concerns
evaluation and is answered with tests chosen to expose optimism.

## 2. Data

The California Housing data describe 20,640 census block groups from the 1990 U.S. census (Pace & Barry, 1997). A
block group typically holds 600 to 3,000 people. Each row gives eight numeric features (median income, median house
age, average rooms, bedrooms and occupants per household, population, latitude and longitude) and the target, the
median house value in units of $100,000 (scikit-learn developers, n.d.). Four properties of the data shaped the design.

- **Scale.** The features differ by orders of magnitude: population runs into the thousands while bedrooms per
  household sit near one. Unscaled, that makes gradient descent badly conditioned (LeCun et al., 2012).
- **Heavy tails.** Three per-household averages and the population count have extreme maxima. The documentation of
  the dataset traces such values to block groups with few households and many empty homes, such as vacation resorts
  (scikit-learn developers, n.d.). They look like real places rather than recording errors, so they are kept, and a
  log transform is tested as an option rather than imposed.
- **Censoring.** House values are top-coded at $500,000, and about 5% of block groups sit at that ceiling. A model
  trained on a censored target learns to predict the ceiling for the most expensive areas, so their errors are
  understated (Tobin, 1958). The capped rows are kept and analysed separately.
- **Spatial structure and aggregation.** Neighbouring block groups resemble one another, so a random split places
  near-duplicates on both sides of the train/test line (Roberts et al., 2017). And because the rows are neighbourhood
  aggregates from 1990, the model describes neighbourhoods as they were then, not houses today; reading its patterns
  as statements about individual homes would be an ecological fallacy (Robinson, 1950).

A fuller account of the dataset is in [DATASHEET.md](DATASHEET.md).

## 3. Splitting and preprocessing

The rows are split once into 70% training, 15% validation and 15% test with a fixed seed, and the three sets are
checked for comparable target distributions with two-sample Kolmogorov–Smirnov statistics (Massey, 1951). Each feature
is standardised with the mean and standard deviation of the training rows only. Fitting the scaler on all rows would
let the validation and test rows shape the training pipeline, which is data leakage (Kaufman et al., 2012). The
notebook measures how far the statistics would move, so the argument rests on a measurement as well as a principle.

Each set has a single job. The training rows fit weights. The validation rows choose epochs, widths, activations and
hyperparameters, and calibrate the prediction intervals. The test rows are scored once, after every decision, and
never feed back into a choice (Hastie et al., 2009).

## 4. The fixed protocol

Every core model has eight standardised inputs, one hidden layer and a linear output unit. It is trained with Adam at
its default learning rate (Kingma & Ba, 2014) on mean squared error for 100 epochs, with a batch size of 64 and
Keras's default Glorot initialisation (Glorot & Bengio, 2010). One hidden layer is enough in principle, because a
wide enough layer can approximate any continuous function on a bounded domain (Hornik et al., 1989). The theorem is
silent, though, about how many units are needed and how well they train, and that is exactly what the experiments
measure. The batch size stays at 64 everywhere, including the tuning extension, so it never becomes a hidden second
variable; small batches also tend to generalise better than very large ones (Keskar et al., 2017).

Training never stops early in the core experiments. A callback records the weights from the epoch with the lowest
validation loss, so the full curves remain visible for diagnosis while the best weights are kept for later use, which
amounts to early stopping with the stopping point read off afterwards (Prechelt, 2012).

## 5. Controlled experiments

**Width.** The network is trained with 16, 64 and 256 hidden units (161, 641 and 2,561 parameters), with ReLU and
everything else fixed. Width sets capacity: too little gives high training and validation error, more fits the
training data more closely, and enough of it opens a gap between the two (Goodfellow et al., 2016).

**Activation.** ReLU, tanh and sigmoid are compared at the selected width, again with everything else fixed. Theory
predicts that sigmoid will be the slowest. Its largest derivative is 0.25 and its outputs are always positive, which
slows gradient descent relative to the zero-centred tanh (LeCun et al., 2012), and saturated sigmoid units pass back
almost no gradient (Glorot & Bengio, 2010). ReLU does not saturate for positive inputs, although a unit that stays
inactive for every example stops learning (Glorot et al., 2011; Nair & Hinton, 2010). To test this rather than assert
it, the notebook measures convergence (epochs to a common validation target, and to 95% of each model's own
improvement), stability (the spread of late-training changes in validation loss) and saturation (the share of trained
hidden pre-activations on the flat part of each curve).

## 6. Seeds and the selection rule

A single training run is one draw from a noisy process: the initial weights and the batch order alone can reorder two
configurations (Bouthillier et al., 2021). Every configuration is therefore trained with five seeds, and selections use
the mean best validation loss. When a simpler option is within one standard error of the best mean, the simpler option
wins; this is the one-standard-error rule (Hastie et al., 2009). Both rules were fixed before any training.

## 7. Tuning beyond the core protocol

A random search then asks how much better a shallow network can get when its remaining settings are tuned. The search
covers the width (32 to 512 units), the activation (ReLU, ELU, GELU or tanh; Clevert et al., 2016; Hendrycks & Gimpel,
2016), Adam's learning rate, L2 weight decay (Krogh & Hertz, 1992), dropout (Srivastava et al., 2014) and a log
transform of the heavy-tailed features. Random sampling covers each setting more densely than a grid of the same size
(Bergstra & Bengio, 2012). Every trial uses early stopping and a learning-rate schedule, and trials are compared on
validation MSE without the weight-decay penalty. A first "anchor" trial re-trains the core configuration under this
protocol, which separates the gain due to the protocol from the gain due to the hyperparameters. Because picking the
best of 25 noisy scores rewards luck as well as quality, the top three configurations are re-trained on two more seeds
before a champion is named (Cawley & Talbot, 2010).

## 8. One test evaluation

With every decision frozen, the selected model, using its best-epoch weights, is evaluated on the test set once. In the
same step, and for context only, the tuned champion and three reference models fitted on the training rows are scored:
a mean predictor, linear regression, and histogram-based gradient-boosted trees in the style of LightGBM (Ke et al.,
2017), a family that is hard to beat on tabular data of this size (Grinsztajn et al., 2022).

## 9. Stronger testing

A single test score hides its own uncertainty, and it flatters a model whose test rows have close neighbours in the
training set. Four further tests address these weaknesses.

- **Bootstrap intervals and paired comparisons.** The test rows are resampled 2,000 times. Percentile intervals show
  how much each metric would move with another plausible test sample, and scoring every model on the same resamples
  gives a paired interval for each difference between models (Efron & Tibshirani, 1993).
- **Random against spatial cross-validation.** The selected configuration is re-trained five times on the training and
  validation rows. Random folds give a conventional estimate; spatial folds hold out whole regions, defined by k-means
  clustering of the coordinates (Lloyd, 1982), so every fold asks the network to price places it has never seen. The
  gap between the two estimates measures the optimism of a random split (Kohavi, 1995; Roberts et al., 2017). The test
  set is not touched.
- **Slice tests.** Errors are broken down by nearest major city, income band, the price ceiling and input extremity,
  each with a bootstrap interval, to find out who bears the largest errors.
- **Conformal prediction intervals**, described in the next section.

## 10. Uncertainty for every prediction

Split conformal prediction turns absolute validation residuals into an interval around each new prediction. If the
calibration and test rows are exchangeable, the interval covers the true value with at least the chosen probability on
average, whatever the error distribution (Vovk et al., 2005; Lei et al., 2018; Angelopoulos & Bates, 2023). Two caveats
are stated rather than hidden. The validation set also chose the model, so the guarantee is approximate, and the test
set measures how close it comes. And the guarantee is marginal: it holds on average over block groups, not within every
group, so coverage is reported separately at the price ceiling, where it is expected to fall short.

## 11. Explanation

Three complementary tools describe what the model relies on. Permutation importance shuffles one feature at a time and
records the rise in test error (Breiman, 2001; Fisher et al., 2019). Partial dependence traces the average prediction
as one feature varies (Friedman, 2001). For a single prediction, exact Shapley values divide the difference between it
and the prediction for a typical block group among the features (Shapley, 1953; Lundberg & Lee, 2017), using the
training medians as one reference point (Sundararajan & Najmi, 2020). With seven players (latitude and longitude count
as one, location), exact computation takes 128 forward passes, so no sampling approximation is needed. All three tools
describe the model, not the housing market; none of them is a causal estimate.

## 12. Pre-registered hypotheses

The hypotheses below, and the rule that decides each one, were written before the final training run.
[RESULTS.md](RESULTS.md) reports every verdict as computed from the artifacts, so a hypothesis can fail in public.

| | Hypothesis | Reason to expect it | Decision rule |
|---|---|---|---|
| H1 | 16 hidden units underfit relative to 64 | limited capacity (Goodfellow et al., 2016) | mean best validation MSE at 16 exceeds that at 64 by more than one standard error |
| H2 | Sigmoid converges more slowly than ReLU and tanh | small, non-zero-centred gradients (LeCun et al., 2012) | more epochs to the common validation target, or never reaching it |
| H3 | The selected model's validation error is optimistic | selection bias (Cawley & Talbot, 2010) | test MSE above the best validation MSE |
| H4 | Unseen regions are harder than random folds suggest | spatial autocorrelation (Roberts et al., 2017) | spatial-fold MAE above random-fold MAE |
| H5 | The network beats linear regression | non-linear effects of income and location | the paired 95% interval for the MAE difference lies below zero |
| H6 | Gradient-boosted trees are at least as accurate | the strength of trees on tabular data (Grinsztajn et al., 2022) | the network is not clearly better in the paired comparison |
| H7 | 90% conformal intervals keep their promise on the test set | the conformal guarantee (Lei et al., 2018) | test coverage within two percentage points of 90% |
| H8 | Errors are larger at the price ceiling | censoring (Tobin, 1958) | test MAE for capped block groups above that for the rest |

## 13. Reproducibility

One seed controls the split, the initial weights and the batch order, and deterministic kernels are switched on, so
re-running a configuration reproduces its curve exactly; the notebook checks this by re-training the baseline and
comparing it epoch by epoch. Finished runs are cached under a hash of their full configuration. The notebook is built
from plain-text sources so that every change can be reviewed as text, which keeps the convenience of the notebook
format (Kluyver et al., 2016) without its opaque diffs. The exported artifacts carry SHA-256 checksums (National
Institute of Standards and Technology, 2015), and every report states the fingerprint of the files it was computed
from. These habits follow the reproducibility checklist that machine-learning venues now ask of authors (Pineau et al.,
2021).

## 14. Conclusion: what the evidence can and cannot establish

The design supports three kinds of claim. Claims about **learning dynamics**, such as how width and activation change
underfitting, convergence and validation error, rest on controlled and seed-replicated comparisons, so they hold for
these data under this protocol. Claims about **performance** rest on a single test evaluation made after every choice,
wrapped in bootstrap intervals and paired comparisons and checked against spatial cross-validation; the reader learns
both the number and how much to trust it. Claims about **uncertainty** rest on conformal intervals whose coverage is
measured on rows the model never saw. The verdicts on H1 to H8, and the numbers behind them, are in
[RESULTS.md](RESULTS.md) together with the fingerprint of the artifacts that produced them, and anyone can recompute
them with `python -m housing_app report`.

The design cannot support claims beyond its scope. It says nothing about house prices today, about individual homes,
about other housing markets or about deeper networks, and its explanations describe the model rather than causes in
the market.

## References

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591. https://doi.org/10.1561/2200000101

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html

Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*, 747–769. https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html

Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2016). *Fast and accurate deep network learning by exponential linear units (ELUs)* [Conference paper]. International Conference on Learning Representations, San Juan, Puerto Rico. https://doi.org/10.48550/arXiv.1511.07289

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593

Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research, 20*(177), 1–81. https://jmlr.org/papers/v20/18-760.html

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics* (pp. 249–256). PMLR. https://proceedings.mlr.press/v9/glorot10a.html

Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics* (pp. 315–323). PMLR. https://proceedings.mlr.press/v15/glorot11a.html

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

Hendrycks, D., & Gimpel, K. (2016). *Gaussian error linear units (GELUs)* (arXiv:1606.08415). arXiv. https://doi.org/10.48550/arXiv.1606.08415

Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks, 2*(5), 359–366. https://doi.org/10.1016/0893-6080(89)90020-8

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154). Curran Associates.

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). *On large-batch training for deep learning: Generalization gap and sharp minima* [Conference paper]. International Conference on Learning Representations, Toulon, France. https://doi.org/10.48550/arXiv.1609.04836

Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization* (arXiv:1412.6980). arXiv. https://doi.org/10.48550/arXiv.1412.6980

Kluyver, T., Ragan-Kelley, B., Pérez, F., Granger, B., Bussonnier, M., Frederic, J., Kelley, K., Hamrick, J., Grout, J., Corlay, S., Ivanov, P., Avila, D., Abdalla, S., Willing, C., & Jupyter Development Team. (2016). Jupyter Notebooks—A publishing format for reproducible computational workflows. In F. Loizides & B. Schmidt (Eds.), *Positioning and power in academic publishing: Players, agents and agendas* (pp. 87–90). IOS Press. https://doi.org/10.3233/978-1-61499-649-1-87

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In *Proceedings of the 14th International Joint Conference on Artificial Intelligence* (Vol. 2, pp. 1137–1143). Morgan Kaufmann.

Krogh, A., & Hertz, J. A. (1992). A simple weight decay can improve generalization. In J. E. Moody, S. J. Hanson, & R. P. Lippmann (Eds.), *Advances in Neural Information Processing Systems 4* (pp. 950–957). Morgan Kaufmann.

LeCun, Y. A., Bottou, L., Orr, G. B., & Müller, K.-R. (2012). Efficient BackProp. In G. Montavon, G. B. Orr, & K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (2nd ed., pp. 9–48). Springer. https://doi.org/10.1007/978-3-642-35289-8_3

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association, 113*(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116

Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory, 28*(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates.

Massey, F. J., Jr. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association, 46*(253), 68–78. https://doi.org/10.1080/01621459.1951.10500769

Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In *Proceedings of the 27th International Conference on Machine Learning* (pp. 807–814). Omnipress.

National Institute of Standards and Technology. (2015). *Secure hash standard (SHS)* (FIPS PUB 180-4). U.S. Department of Commerce. https://doi.org/10.6028/NIST.FIPS.180-4

Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters, 33*(3), 291–297. https://doi.org/10.1016/S0167-7152(96)00140-X

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research (A report from the NeurIPS 2019 reproducibility program). *Journal of Machine Learning Research, 22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html

Prechelt, L. (2012). Early stopping—But when? In G. Montavon, G. B. Orr, & K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (2nd ed., pp. 53–67). Springer. https://doi.org/10.1007/978-3-642-35289-8_5

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review, 15*(3), 351–357. https://doi.org/10.2307/2087176

scikit-learn developers. (n.d.). *California Housing dataset*. In *scikit-learn user guide*. Retrieved October 5, 2026, from https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the theory of games* (Vol. 2, pp. 307–317). Princeton University Press.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Sundararajan, M., & Najmi, A. (2020). The many Shapley values for model explanation. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 9269–9278). PMLR. https://proceedings.mlr.press/v119/sundararajan20b.html

Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica, 26*(1), 24–36. https://doi.org/10.2307/1907382

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer. https://doi.org/10.1007/b106715
