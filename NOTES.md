# Personal Notes: Tucker vs. CP Decomposition

## Core intuition

A tensor decomposition is compression by finding *structure* — instead of
storing every number in the tensor, store a small set of "building blocks"
that can reconstruct it approximately.

- **Tucker** = SVD generalized to 3+ modes, with a small dense core encoding
  cross-mode interactions.
- **CP** = Tucker with the core forced diagonal → sum of rank-1 terms, no
  cross-mode interaction beyond the sum itself.

## What "rank" means differently in each method

- Tucker: **one rank per mode** — `(r1, r2, r3)`. You can compress height
  and width aggressively while keeping all 3 color channels.
- CP: **one rank total** — `R` rank-1 components, each spanning all modes
  simultaneously. Less flexible per-mode, but often fewer total parameters
  for the same rank number.

This is why comparing "Tucker rank 100" to "CP rank 100" isn't a perfectly
fair fight — they're not the same kind of number.

## What I found

- At roughly comparable settings, Tucker error = 0.0931, CP error = 0.1016.
  Tucker's dense core buys it a real, if modest, accuracy edge.
- CP still edges out Tucker on compression ratio (14.10x vs 12.32x) despite
  the higher error — fewer parameters per rank when there's no core tensor.
- In the Tucker factor breakdown, **Factor 0 and Factor 1 (height, width)
  dominate storage size** — the channel factor and core are comparatively
  tiny. Compression gains come almost entirely from truncating spatial rank,
  not channel rank.
- CP's ALS optimization is noticeably slower than Tucker's direct HOSVD,
  especially as rank grows — HOSVD is a closed-form SVD computation, ALS is
  iterative and has to converge.

## Open questions / things to explore next

- [ ] Does CP's `random_state` (initialization) meaningfully change results
      at the same rank? Try 2-3 different seeds and compare.
- [ ] Tucker's non-uniqueness: rotate the factor matrices and core, confirm
      reconstruction is identical (per the original lab slide's bullet point).
- [ ] Try a smoother/low-detail image and see whether the gap between
      Tucker and CP narrows or widens — my current image (high-frequency
      abstract art) is a stress test, not a typical case.
- [ ] Read Dr. Behera's papers on tensor methods before the internship —
      check whether NATL Lab's applications favor Tucker or CP-style
      decompositions and why.

## Vocabulary I want to keep straight

- **Fiber**: 1D slice of a tensor (fix all but one index)
- **Unfolding/matricization**: reshaping a tensor into a matrix along one mode
- **Mode-n product (×ₙ)**: generalization of matrix multiplication for tensors
- **Multilinear rank**: the tuple of ranks `(r1, r2, ..., rN)` for Tucker
- **Relative Frobenius error**: `||X - X_hat|| / ||X||`, normalized reconstruction error