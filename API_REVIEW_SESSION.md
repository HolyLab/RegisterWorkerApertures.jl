# Session Handoff — 2026-05-20

## Plan
API_REVIEW_PLAN.md — RegisterWorkerApertures v1.0.0

## What was just completed
**CHUNK-006: version-bump**
Bumped `Project.toml` from `0.2.4` → `1.0.0`. No CHANGELOG.md exists. Full test suite passes at v1.0.0. This is the terminal release — all six chunks are now complete or satisfied.

## Key decisions / shim choices
- n/a (version bump only)

## State of the codebase
- Files modified: `Project.toml` (staged, not yet committed)
- Test suite: pass (v1.0.0 shown in test output header)
- Ambiguity count: 0
- Staged but uncommitted: yes — `Project.toml`

## Cluster status
- `baseline-blocker`: 1 of 1 complete ✓
- `dead-code-cleanup`: 2 of 2 complete ✓
- `api-surface`: 2 of 2 complete ✓
- All chunks: complete ✓

## Next chunk
None — all chunks complete. The only remaining action is the **release** itself (user action).

## Release checklist (user actions required)
1. Commit `Project.toml` on branch `dwk/mng`.
2. Open a PR and merge to `master`.
3. On the merge commit on GitHub, post: `@JuliaRegistrator register`
4. The Julia registry PR will open automatically; approve/merge it after the normal waiting period.

**Note on RegisterWorkerShell**: The threading bug fix (commit `a437359` on `master`) is not yet registered. If CI needs to resolve RegisterWorkerShell from the registry (not from dev), bump RegisterWorkerShell to `v1.0.2` and register it first, then register this package. Otherwise, the `[compat]` bound `RegisterWorkerShell = "1"` will resolve via dev for local use, and CI may fail until the package is published.

## Watch out for
- Registration on the Julia registry is **separate** from the git tag. `@JuliaRegistrator` must be commented on the merge commit on GitHub — not just locally tagging.
- The `[compat]` section requires all `Register*` dependencies at `"1"`. If any of those aren't published in the registry yet, CI will fail. All nine were `Pkg.develop`'d locally as a workaround — that's developer-machine-only and won't help CI.
