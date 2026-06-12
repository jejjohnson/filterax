# References

Master reference list for the Mathematical Reference, grouped by
chapter. Per-chapter lists repeat the most load-bearing entries.

## Foundational DA texts

1. Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman
   Filter.* 2nd ed., Springer.
2. Carrassi, A., Bocquet, M., Bertino, L., & Evensen, G. (2018).
   *Data assimilation in the geosciences: An overview of methods,
   issues, and perspectives.* WIREs Climate Change, 9(5), e535.
3. Asch, M., Bocquet, M., & Nodet, M. (2016). *Data Assimilation:
   Methods, Algorithms, and Applications.* SIAM.
4. Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
   formulation and practical implementation.* Ocean Dynamics, 53,
   343–367.

## EnKF — stochastic formulation (chapters 4–5)

5. Evensen, G. (1994). *Sequential data assimilation with a nonlinear
   quasi-geostrophic model using Monte Carlo methods to forecast
   error statistics.* J. Geophys. Res., 99(C5), 10143–10162.
6. Burgers, G., van Leeuwen, P. J., & Evensen, G. (1998). *Analysis
   scheme in the ensemble Kalman filter.* Mon. Wea. Rev., 126,
   1719–1724.

## ETKF and square-root variants (chapters 6–7)

7. Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001).
   *Adaptive sampling with the ensemble transform Kalman filter.
   Part I: Theoretical aspects.* Mon. Wea. Rev., 129, 420–436.
8. Wang, X., Bishop, C. H., & Julier, S. J. (2004). *Which is better,
   an ensemble of positive–negative pairs or a centered spherical
   simplex ensemble?* Mon. Wea. Rev., 132, 1590–1605. (Square-root
   choices; the symmetric square root.)
9. Livings, D. M., Dance, S. L., & Nichols, N. K. (2008). *Unbiased
   ensemble square root filters.* Physica D, 237(8), 1021–1028.
10. Sakov, P. & Oke, P. R. (2008). *Implications of the form of the
    ensemble transformation in the ensemble square root filters.*
    Mon. Wea. Rev., 136, 1042–1053.
11. Whitaker, J. S. & Hamill, T. M. (2002). *Ensemble data
    assimilation without perturbed observations.* Mon. Wea. Rev.,
    130, 1913–1924. (EnSRF / serial assimilation.)
12. Nerger, L., Janjić, T., Schröter, J., & Hiller, W. (2012). *A
    unification of ensemble square root Kalman filters.* Mon. Wea.
    Rev., 140, 2335–2345. (ESTKF.)
13. Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., &
    Whitaker, J. S. (2003). *Ensemble square root filters.* Mon.
    Wea. Rev., 131, 1485–1490.

## LETKF (chapter 8)

14. Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient
    data assimilation for spatiotemporal chaos: A local ensemble
    transform Kalman filter.* Physica D, 230, 112–126.
15. Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich,
    E. J., Corazza, M., Kalnay, E., Patil, D. J., & Yorke, J. A.
    (2004). *A local ensemble Kalman filter for atmospheric data
    assimilation.* Tellus A, 56(5), 415–428.

## Ensemble Kalman processes — EKI / EKS / UKI (chapter 9)

16. Iglesias, M. A., Law, K. J. H., & Stuart, A. M. (2013).
    *Ensemble Kalman methods for inverse problems.* Inverse
    Problems, 29(4), 045001.
17. Garbuno-Inigo, A., Hoffmann, F., Li, W., & Stuart, A. M. (2020).
    *Interacting Langevin diffusions: Gradient structure and ensemble
    Kalman sampler.* SIAM J. Appl. Dyn. Syst., 19(1), 412–441.
18. Huang, D. Z., Schneider, T., & Stuart, A. M. (2022). *Iterated
    Kalman methodology for inverse problems.* J. Comput. Phys., 463,
    111262. (UKI.)
19. Kovachki, N. B. & Stuart, A. M. (2019). *Ensemble Kalman
    inversion: a derivative-free technique for machine learning
    tasks.* Inverse Problems, 35(9), 095005.
20. Iglesias, M. A. (2016). *A regularizing iterative ensemble Kalman
    method for PDE-constrained inverse problems.* Inverse Problems,
    32(2), 025002. (Data-misfit step-size control.)

## Smoothers (chapter 10)

21. Evensen, G. & van Leeuwen, P. J. (2000). *An ensemble Kalman
    smoother for nonlinear dynamics.* Mon. Wea. Rev., 128,
    1852–1867.
22. Cosme, E., Verron, J., Brasseur, P., Blum, J., & Auroux, D.
    (2012). *Smoothing problems in a Bayesian framework and their
    linear Gaussian solutions.* Mon. Wea. Rev., 140, 683–695.
23. Chen, Y. & Oliver, D. S. (2013). *Levenberg–Marquardt forms of
    the iterative ensemble smoother for efficient history matching
    and uncertainty quantification.* Comput. Geosci., 17, 689–703.
    (IES.)
24. Bocquet, M. & Sakov, P. (2014). *An iterative ensemble Kalman
    smoother.* Q. J. R. Meteorol. Soc., 140(682), 1521–1535.
25. Whitaker, J. S. & Compo, G. P. (2002). *An ensemble Kalman
    smoother for reanalysis.* Proc. Symp. on Observations, Data
    Assimilation and Probabilistic Prediction, AMS.
26. Evensen, G., Raanes, P. N., Stordal, A. S., & Hove, J. (2019).
    *Efficient implementation of an iterative ensemble smoother for
    data assimilation and reservoir characterization.* Front. Appl.
    Math. Stat., 5, 47.

## Localization (chapter 11)

27. Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
    functions in two and three dimensions.* Q. J. R. Meteorol. Soc.,
    125, 723–757.
28. Houtekamer, P. L. & Mitchell, H. L. (2001). *A sequential
    ensemble Kalman filter for atmospheric data assimilation.* Mon.
    Wea. Rev., 129, 123–137.
29. Anderson, J. L. (2007). *Exploring the need for localization in
    ensemble data assimilation using a hierarchical ensemble
    filter.* Physica D, 230, 99–111.
30. Thiebaux, H. J. & Pedder, M. A. (1987). *Spatial Objective
    Analysis.* Academic Press. (SOAR correlation model.)

## Inflation (chapter 12)

31. Anderson, J. L. & Anderson, S. L. (1999). *A Monte Carlo
    implementation of the nonlinear filtering problem to produce
    ensemble assimilations and forecasts.* Mon. Wea. Rev., 127,
    2741–2758.
32. Zhang, F., Snyder, C., & Sun, J. (2004). *Impacts of initial
    estimate and observation availability on convective-scale data
    assimilation.* Mon. Wea. Rev., 132, 1238–1253. (RTPP.)
33. Whitaker, J. S. & Hamill, T. M. (2012). *Evaluating methods to
    account for system errors in ensemble data assimilation.* Mon.
    Wea. Rev., 140, 3078–3089. (RTPS.)
34. Anderson, J. L. (2009). *Spatially and temporally varying
    adaptive covariance inflation for ensemble filters.* Tellus A,
    61, 72–83.
35. Hamill, T. M. & Whitaker, J. S. (2005). *Accounting for the
    error due to unresolved scales in ensemble data assimilation: A
    comparison of different approaches.* Mon. Wea. Rev., 133,
    3132–3147. (Additive inflation.)
36. Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for
    large-dimensional covariance matrices.* J. Multivariate Anal.,
    88, 365–411.

## Diagnostics (chapter 13)

37. Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005).
    *Diagnosis of observation, background and analysis-error
    statistics in observation space.* Q. J. R. Meteorol. Soc.,
    131(613), 3385–3396.
38. Hersbach, H. (2000). *Decomposition of the Continuous Ranked
    Probability Score for Ensemble Prediction Systems.* Wea.
    Forecasting, 15(5), 559–570.
39. Hamill, T. M. (2001). *Interpretation of Rank Histograms for
    Verifying Ensemble Forecasts.* Mon. Wea. Rev., 129(3), 550–560.
40. Mehra, R. K. (1970). *On the Identification of Variances and
    Adaptive Kalman Filtering.* IEEE Trans. Automatic Control,
    15(2), 175–184.
41. Cardinali, C., Pezzulli, S., & Andersson, E. (2004).
    *Influence-Matrix Diagnostic of a Data Assimilation System.*
    Q. J. R. Meteorol. Soc., 130(603), 2767–2786.
42. Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring
    Rules, Prediction, and Estimation.* JASA, 102(477), 359–378.
43. Fortin, V., Abaza, M., Anctil, F., & Turcotte, R. (2014). *Why
    Should Ensemble Spread Match the RMSE of the Ensemble Mean?*
    J. Hydrometeorol., 15(4), 1708–1713.
44. Whitaker, J. S. & Loughe, A. F. (1998). *The Relationship
    between Ensemble Spread and Ensemble Mean Skill.* Mon. Wea.
    Rev., 126(12), 3292–3302.
45. Liu, J. S. & Chen, R. (1998). *Sequential Monte Carlo Methods
    for Dynamic Systems.* JASA, 93(443), 1032–1044.

## Differentiable & latent DA (chapters 14–15)

46. Chen, Y., Sanz-Alonso, D., & Willett, R. (2023). *Reduced-Order
    Autodifferentiable Ensemble Kalman Filters.* Inverse Problems,
    39(12), 124001. (ROAD-EnKF.)
47. Griewank, A. & Walther, A. (2000). *Algorithm 799: revolve — an
    implementation of checkpointing for the reverse or adjoint mode
    of computational differentiation.* ACM TOMS, 26(1), 19–45.
48. Peyron, M., Fillion, A., Gürol, S., Marchais, V., Gratton, S.,
    Boudier, P., & Goret, G. (2021). *Latent space data assimilation
    by using deep learning.* Q. J. R. Meteorol. Soc., 147(740),
    3759–3777.
49. Amendola, M., Arcucci, R., Mottet, L., Casas, C. Q., Fan, S.,
    Pain, C., Linden, P., & Guo, Y.-K. (2021). *Data assimilation in
    the latent space of a convolutional autoencoder.* ICCS 2021,
    LNCS 12746.

## Ecosystem libraries

- [`gaussx`](https://github.com/jejjohnson/gaussx) — structured
  linear operators; the Woodbury / matrix-determinant-lemma dispatch
  behind the gain and likelihood, the Gaspari-Cohn taper, and the
  classic inflation trio.
- [`lineax`](https://github.com/patrick-kidger/lineax) — linear
  operators and solvers (`AbstractLinearOperator` for $R$, $Q$).
- [`equinox`](https://github.com/patrick-kidger/equinox) — module
  system; every filterax component is an `eqx.Module` PyTree.
- [`optax`](https://github.com/google-deepmind/optax) — optimizers
  for the differentiable-training surface.
- [`pipekit`](https://github.com/jejjohnson/pipekit) +
  `pipekit-cycle` — operator composition and the DA-cycle protocols
  of chapter 16.
- `vardax` — variational DA sibling library (OI / 3DVar / 4DVar);
  `filterax` provides its ensemble-posterior bridge.
