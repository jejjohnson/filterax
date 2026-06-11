# Localization

A small ensemble (`N_e ≪ N_x`) produces a rank-deficient sample covariance
whose long-range entries are dominated by sampling noise: two physically
unrelated variables will show spurious correlation of order `1/√N_e`, and the
filter will happily use it to spread observation increments across the whole
domain. Localization suppresses these spurious long-range correlations by
damping covariance entries as a function of physical distance.

The standard mechanism is the Schur (element-wise) product `P_loc = ρ ∘ P`
with a taper matrix `ρ` built from a compactly supported correlation
function. By the Schur product theorem, the element-wise product of two
positive semi-definite matrices is positive semi-definite — so as long as the
taper is itself a valid correlation matrix (Gaspari-Cohn and SOAR are; a hard
cutoff is not, in general), localization cannot break the PSD-ness of the
covariance.

[`gaspari_cohn`][filterax.gaspari_cohn] and
[`localization_matrix`][filterax.localization_matrix] delegate to their
gaussx counterparts with support parameter `c = 2 * radius` (gaussx
parameterises by the compact-support length, filterax by the half-width);
the distance metrics ([`euclidean_distance`][filterax.euclidean_distance],
[`haversine_distance`][filterax.haversine_distance]) are re-exported from
gaussx unchanged. Wrap these primitives in an
[`AbstractLocalizer`][filterax.AbstractLocalizer] to plug them into the
Layer-2 loops as a drop-in component; the
[`LETKF`][filterax.filters.LETKF] instead applies localization in
observation space, per grid point, and
[`localized_kalman_gain`][filterax.localized_kalman_gain] tapers the gain
itself.

## Taper functions

Map a distance array to weights in `[0, 1]`. Gaspari-Cohn is the
production default (compactly supported, exactly zero beyond `2 * radius`);
Gaussian and SOAR decay smoothly without compact support; the hard cutoff is
a diagnostic tool rather than a valid correlation function.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [gaspari_cohn, gaussian_taper, soar_taper, hard_cutoff]

## Distances & taper matrices

[`localization_matrix`][filterax.localization_matrix] fuses a pairwise
distance computation with the Gaspari-Cohn taper, producing the dense
`ρ` consumed by [`localize`][filterax.localize] and
[`localized_kalman_gain`][filterax.localized_kalman_gain]. Use
[`haversine_distance`][filterax.haversine_distance] as the metric for
spherical (lat, lon)-in-radians grids.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [localization_matrix, euclidean_distance, haversine_distance]

## Applying localization

[`localize`][filterax.localize] is the Schur product itself;
[`adaptive_localization`][filterax.adaptive_localization] builds the weight
matrix from the ensemble's own sampling-error statistics (Anderson 2007)
instead of a prescribed distance taper — useful when no meaningful distance
metric exists (e.g. parameter spaces).

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [localize, localized_kalman_gain, adaptive_localization]
