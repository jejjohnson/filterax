# Agent Guidelines

This file contains standing instructions for **all** coding agents working on this repository (Copilot, Claude, Gemini, etc.).

---

## Karpathy Coding Principles

Four behavioral principles to reduce the most common LLM coding mistakes. These bias toward caution over speed — for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State assumptions explicitly. If uncertain, ask before writing code.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Test: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into declarative goals with verification:

- "Add validation" → Write tests for invalid inputs, then make them pass
- "Fix the bug" → Write a test that reproduces it, then make it pass
- "Optimize X" → Write the naive correct version first, then optimize while preserving correctness

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Before Every Commit

**All agents must** verify that every one of the following passes before creating a commit or reporting progress. No exceptions.

1. **Tests** – `uv run pytest -q` (or `make test`) must have 0 failures.
2. **Lint** – `uv run --group lint ruff check .` (or `make lint`) must report no issues.
3. **Format** – `uv run --group lint ruff format --check .` must report no files to reformat.
4. **Type checks** – `uv run --group typecheck ty check src/filterax` (or `make typecheck`) must report no errors in changed files.

### Test tiers

CI on every PR runs the fast tier (`-m "not slow and not integration"`, or `make test-fast`) on Python 3.12 and 3.13.
The "Extended Tests" workflow runs the entire suite weekly, the heavy lane (`make test-slow`) on PRs labelled `run-slow`, and either on demand.
Add `run-slow` to a PR that touches the numerics of a filter, smoother or process.

- **Unmarked:** unit tests of about 1 s or less.
- **`@pytest.mark.slow`:** over about 1.5 s: heavy numerics, `jit`+`grad`+`vmap` sweeps, long scans.
- **`@pytest.mark.integration`:** end-to-end workflows, meaning a full assimilation, inversion or training loop through the public API, whatever its duration.

Most runtime is XLA compiling each operation the first time, so a test can look cheap only because an earlier test paid for its compile. Judge a tier with `pytest -m "not slow and not integration" --durations=20`, not by the test on its own.

> **Common pitfall**: Running `ruff check src/filterax/` instead of `ruff check .` misses lint errors in `tests/` and `scripts/`. CI runs `ruff check .` on the entire repo — always use `.` (repo root), not a subdirectory.

## Development Environment

**IMPORTANT**: Always use `uv run` when running Python tools or scripts (e.g., `pytest`, `python`, `ruff`, `ty`, `mkdocs`, `pre-commit`) so they run in the project environment. You do **not** need `uv run` for non-Python shell commands (e.g., `git`, `ls`, `cat`). Do NOT use the system Python directly.

## Pull Request Descriptions

**Never replace or remove an existing PR title or description.** When reporting progress on a PR that already has a title and description, only append new checklist items or update the status of existing ones. The original content must be preserved in full.

This is a common failure mode: an agent called to make a small follow-up change will supply a fresh description scoped only to its own work, silently discarding all prior context. Always read the existing description first and treat it as the base.

## PR Title & Description Rules

**Do not change** the PR title or description between sessions except to:
- Correct a conventional-commits format violation in the title.
- Append new items to the description checklist.

Never rewrite the existing description; only add to it.

## Pull Request Review Comments

When addressing PR review comments, **resolve each review thread** after fixing it. Use the GitHub GraphQL API to list threads and resolve them:

```bash
# 1. List all review threads and their IDs for a PR
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes {
            id
            isResolved
            comments(first: 1) {
              nodes { body path line }
            }
          }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve a review thread by its node ID
gh api graphql -f query='mutation($threadId: ID!) { resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' -f threadId=THREAD_ID
```

Workflow:
1. List review threads via the GraphQL query above to get thread node IDs
2. Address each thread with code changes
3. After pushing the fix, resolve each addressed review thread via the `resolveReviewThread` mutation
4. Do **not** resolve threads you haven't addressed

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). If a plan needs to be tracked long-term, create a GitHub issue with the same detail instead. **Never commit plan files to the repository.**

## GIT Safety Rules

- **NEVER** push to `main` or merge into `main` unless the user explicitly says "push to main" or "merge to main".
- **NEVER** push to any remote branch or run `git push` unless the user explicitly asks you to push. Only commit locally.
- Always work on feature branches.
- When the user says "merge the changes" or "merge the branch", they mean push the local branch to the remote — NOT merge into main.
- Always confirm before any action that affects shared branches (main, production, etc.).

## Documentation

This repo uses **MkDocs + Material + mkdocstrings + mkdocs-jupyter** for documentation.

- **Build locally**: `make docs-serve` (or `uv run --group docs mkdocs serve`)
- **Build static site**: `make docs` (or `uv run --group docs mkdocs build`)
- **Deploy to GitHub Pages**: `make docs-deploy` (or `uv run --group docs mkdocs gh-deploy --force`)
- **Auto-deploy**: the `pages.yml` workflow deploys automatically on every push to `main`

When writing docstrings, use **Google style** (enforced by `mkdocstrings` config).

Notebooks in `docs/` may be stored as `.ipynb` files or as `jupytext`-paired `.py` files.

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
- `feat: add new data loading utility`
- `fix: correct off-by-one error in slice computation`
- `docs: update installation instructions`
- `chore: bump ruff to 0.9.7`
