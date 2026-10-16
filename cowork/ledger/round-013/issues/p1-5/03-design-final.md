# Design P1-5 v1 (user-locked)
## Goal
SuperPoint keypoint sort uses a strict weak ordering.
## Bounds
One-line comparator; no score-threshold semantics change.
## Root cause
`return pt1.score >= pt2.score` is reflexive → UB for std::sort.
## Fix
Use `>` (descending by score).
