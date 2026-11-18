# Project workflow

- Commit every completed, verified logical change to Git. The user explicitly
  requested ongoing commits; do not leave completed changes uncommitted.
- Use descriptive commit messages and run checks appropriate to the change.
- Keep photos, edited images, generated QA output, build packages, dependencies,
  model weights and secrets out of Git. Keep model manifests and licenses tracked.
- Preserve original photos and existing user sessions during development.
- Do not rewrite existing commit history unless the user explicitly requests it.
- When an authorized GitHub remote is available, keep completed commits pushed
  to that remote. Report authentication or network failures honestly.
