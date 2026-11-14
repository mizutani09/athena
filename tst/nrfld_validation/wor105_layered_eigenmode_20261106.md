# WOR-105: exact two-material AMR diffusion eigenmode

## Analytic problem

This test isolates the face coefficient at an optically thick material jump
aligned with the AMR boundary. The periodic domain is `0<=x<1`, with
`rho_outer=1` outside `[0.25,0.75)` and `rho_inner=r` inside.
`sigma=kappa*rho`, `D=c/(3*sigma)`, and `lambda=1/3` remains fixed.

On `0<=x<=0.25`, set `E=1+A*cos(k*x)`. On `0.25<=x<=0.5`,
set `E=1+A*B*cos(k*sqrt(r)*(0.5-x))`, then reflect about `x=0.5`.
The first positive root of

`sin(k/4)*cos(k*sqrt(r)/4) + cos(k/4)*sin(k*sqrt(r)/4)/sqrt(r) = 0`

and the corresponding `B` make both `E` and `D*dE/dx` continuous
at the material boundary. The Python driver checks both conditions.
The decay eigenvalue is `mu=D_outer*k*k`. The problem generator writes
exact cell averages on both AMR levels. Each snapshot is compared against
the same backward-Euler temporal factors as the code,
`prod_n (1+mu*dt_n)^(-1)`, so the metric isolates spatial error.
The test requires double-precision HDF5 output and rejects float32.

The transverse domain is `8/N` in each direction. It keeps cubic cells
and one transverse block while exercising 3D AMR exchange. The run uses
`A=0.05`, `kappa=4`, and `t_final=0.003`. The smallest fine-cell
optical depth is 6.25 (ratio 0.1, N128).

## Results

Volume-weighted mean absolute error at the common final time:

| Inner/outer density | Base N | Steps | Legacy | Shared `kavg` | Shared series resistance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 32 | 8 | 3.053e-4 | 3.069e-4 | **6.817e-6** |
| 0.1 | 64 | 16 | 1.431e-4 | 1.349e-4 | **2.221e-6** |
| 0.1 | 128 | 32 | 6.995e-5 | 5.942e-5 | **5.938e-7** |
| 10 | 32 | 2 | 8.615e-5 | 5.040e-5 | **4.627e-7** |
| 10 | 64 | 3 | 5.786e-5 | 3.274e-5 | **1.795e-7** |
| 10 | 128 | 5 | 3.488e-5 | 1.613e-5 | **5.537e-8** |

The N64-to-N128 observed order is 1.18 (shared `kavg`) versus **1.90**
(series resistance) for ratio 0.1, and 1.02 versus **1.70** for ratio 10.
All modes began at zero error against the exact cell averages. Every active
Newton solve strictly converged and kept `E_r>0`. Both shared modes
conserved the periodic integral to about `1e-17` or better, with
rank-summed diffusion ledgers below `4e-19`. The legacy integral drift
at N64 was `+5.78e-7` for ratio 0.1 and `-4.14e-7` for ratio 10.
Four-rank N64 runs reproduced the serial fields and errors for both ratios.

The previous one-step Gaussian comparison against uniform fine output had
mixed rankings, especially for ratio 0.1 at N64. That initial Gaussian
does not satisfy continuous material-interface flux, and its VTK values
were float32. It remains a useful transient stress test, but this exact
eigenmode is a more direct diffusion-limit coefficient test.

## Decision

The shared-flux construction is correct for conservation. The remaining
choice is its coefficient. CASTRO's `kavg` is a same-level face-opacity
rule; its level-local solves reconcile coarse/fine differences through
deferred synchronization. At a composite coarse/fine face, the cell
centers are unequal distances from the face. The physical diffusion
resistance is their distance-weighted sum. The experimental series formula
implements that weighting for the selected two-point ghost, greatly
improving this aligned two-material case without changing conservation.

The series formula is now the preferred **fixed-limiter thick-interface
candidate** for further testing. It remains opt-in: thin-limit capping,
variable limiter, misaligned or smooth coefficients, multidimensional
tangential gradients, and coupled production conditions need separate
evaluation. The limiter remains fixed within each Newton solve.

Reproduce with:

```sh
python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld -mpi -hdf5 -h5double
make -j8
python3 tst/nrfld_validation/wor105_layered_eigenmode.py \
  --binary bin/athena --output /tmp/wor105_layered_check \
  --resolutions 32 64 128 --ratios 0.1 10
```

Serial outputs: `/tmp/wor105_layered_eigen_matrix` (N32 ratio 0.1),
`/tmp/wor105_layered_eigen_smoke2` (N32 ratio 10),
`/tmp/wor105_layered_eigen_n64` (N64 ratio 0.1),
`/tmp/wor105_layered_eigen_n64_dense` (N64 ratio 10),
`/tmp/wor105_layered_eigen_n128` (both N128 ratios).
Four-rank outputs: `/tmp/wor105_layered_eigen_mpi4`.
