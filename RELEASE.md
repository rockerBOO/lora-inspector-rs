# Release Process

## Quick Release
```bash
# Commit any pending changes first
git add <files>
git commit -m "Description of changes"

# Run tests and format
make test && make fmt

# Bump version (creates commit and tag)
./scripts/bump_release.py prerelease  # or major/minor/patch

# Push
git push origin main && git push origin <tag>

# GitHub Actions builds binaries and creates draft release
```

## Version Commands
```bash
./scripts/check_version.py              # Check current version
./scripts/bump_release.py major         # → 2.0.0-beta.1
./scripts/bump_release.py minor         # → 1.1.0-beta.1
./scripts/bump_release.py patch         # → 1.0.1-beta.1
./scripts/bump_release.py prerelease    # → 1.0.0-beta.2
./scripts/bump_release.py release       # → 1.0.0 (remove prerelease)
./scripts/bump_release.py set 1.2.3     # Set specific version
```

## Version Types
- `--pre alpha|beta|rc` to specify prerelease type (default: beta)
- Versions with suffix (e.g., `1.0.0-beta.1`) = prerelease
- All releases are created as drafts