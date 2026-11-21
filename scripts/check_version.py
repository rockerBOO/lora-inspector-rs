#!/usr/bin/env python3
import os
import subprocess
import tomllib
from collections.abc import Sequence

def find_cargo_tomls(root_dir: str = 'crates') -> list[str]:
    """Find all Cargo.toml files in the specified directory."""
    cargo_tomls = []
    for root, _, files in os.walk(root_dir):
        if 'Cargo.toml' in files:
            cargo_tomls.append(os.path.join(root, 'Cargo.toml'))
    return cargo_tomls + ['Cargo.toml']

def get_git_tags() -> list[str]:
    """Retrieve git tags."""
    try:
        tags = subprocess.check_output(['git', 'tag'], text=True).strip().split('\n')
        return tags if tags != [''] else []
    except subprocess.CalledProcessError:
        return []

def get_cargo_version(path: str, root_toml: dict) -> str | None:
    """Read version from a specific Cargo.toml."""
    try:
        with open(path, 'rb') as f:
            cargo_toml = tomllib.load(f)

            # If workspace version is used, get from root Cargo.toml
            if cargo_toml.get('package', {}).get('version') == {'workspace': True}:
                return root_toml.get('workspace', {}).get('package', {}).get('version')

            # Direct package version
            return cargo_toml.get('package', {}).get('version')
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_version(version: str) -> tuple[int, int, int]:
    """Parse version string into major, minor, patch components."""
    # Remove prerelease suffix if present
    base_version = version.split('-')[0]
    parts = base_version.split('.')
    return (int(parts[0]), int(parts[1]), int(parts[2]))

def suggest_next_version(current_version: str, git_history: Sequence[str]) -> str:
    """Suggest next version based on git history and current version."""
    major, minor, patch = parse_version(current_version)

    # Check git log for significant changes
    try:
        # Get commit messages since last tag
        log_output = subprocess.check_output(
            ['git', 'log', '--pretty=format:%s'],
            text=True
        ).lower()

        # Determine version bump based on commit messages
        if any(keyword in log_output for keyword in ['breaking', 'major', 'redesign', 'refactor']):
            return f"{major + 1}.0.0"
        elif any(keyword in log_output for keyword in ['feature', 'new', 'enhancement', 'add']):
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"
    except subprocess.CalledProcessError:
        # Default to patch version if log check fails
        return f"{major}.{minor}.{patch + 1}"

def main():
    # Read root Cargo.toml to get workspace version
    try:
        with open('Cargo.toml', 'rb') as f:
            root_toml = tomllib.load(f)
    except FileNotFoundError:
        print("Could not find root Cargo.toml")
        return

    # Find all Cargo.toml files
    cargo_tomls = find_cargo_tomls()

    # Get git tags
    git_tags = get_git_tags()

    # Collect versions
    versions: dict[str, str] = {}
    for toml_path in cargo_tomls:
        version = get_cargo_version(toml_path, root_toml)
        if version:
            versions[toml_path] = version

    # Print versions
    print("Crate Versions:")
    for path, version in versions.items():
        print(f"  {path.replace('crates/', '')}: {version}")

    print("\nGit Tags:")
    if git_tags:
        for tag in git_tags:
            print(f"  {tag}")
    else:
        print("  No tags found")

    # Suggest next version based on root Cargo.toml
    root_version = root_toml.get('workspace', {}).get('package', {}).get('version')
    if root_version:
        suggested_version = suggest_next_version(root_version, git_tags)
        print(f"\nSuggested next version: {suggested_version}")

        # Compare with existing tags
        version_tags = [tag for tag in git_tags if tag.lstrip('v') == root_version]

        if version_tags:
            print("✓ Version tag matches Cargo.toml version")
        else:
            print(f"! No git tag matching version {root_version}")

if __name__ == '__main__':
    main()