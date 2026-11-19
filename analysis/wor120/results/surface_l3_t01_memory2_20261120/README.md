# Surface-L3 sustained memory2 result

- RK2+x2 XD job: `343498`
- RK4+4wz XD job: `343579`
- physical interval: code time `0--47.006` (`0.1 t_dyn`)
- accepted seed SHA-256:
  `1b1d43db0c2636740c3c41163896455e0c0844d1d240a606cf8a5f17da55237a`
- 4wz executable SHA-256:
  `042fcd1701f4215c038a38ec80746ca7671163ba06472eb8c9aa8219b7e9811e`
- final 4wz restart SHA-256:
  `c3e4249de92293493b2d665a8668625c847d7c9e99def9bc74af7c47b280e389`

The comparison, summary, and gate JSON files are direct outputs of the C++ analyzers
in `analysis/wor120` and `analysis/wor162`.  `virial_trend.json` records the required
least-squares review over the final half of both histories.  See
`doc/wor120_hd_4wz_production_decision_20261120.md` for the interpretation and
production decision.
