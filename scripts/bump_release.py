#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import tomllib

def write_toml(data: dict, path: str) -> None:
    """Write a dictionary to a TOML file manually."""
    def format_value(value):
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, dict):
            return "{" + ", ".join(f"{k} = {format_value(v)}" for k, v in value.items()) + "}"
        return str(value)

    with open(path, 'w') as f:
        for section, content in data.items():
            f.write(f"[{section}]\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"{key} = {format_value(value)}\n")
            f.write("\n")

def read_cargo_toml(path: str) -> str:
    """Read current version from Cargo.toml."""
    try:
        with open(path, 'rb') as f:
            cargo_toml = tomllib.load(f)

            # Check workspace version
            version = (cargo_toml.get('workspace', {})
                       .get('package', {})
                       .get('version'))

            if version:
                return version

            # Fallback to direct package version
            version = cargo_toml.get('package', {}).get('version')

            if version:
                return version

            raise ValueError("No version found in Cargo.toml")
    except (FileNotFoundError, KeyError):
        raise ValueError("No version found in Cargo.toml")

def update_cargo_toml(path: str, new_version: str) -> None:
    """Update version in Cargo.toml."""
    try:
        with open(path, 'rb') as f:
            cargo_toml = tomllib.load(f)

        # Determine if it's a workspace or direct version
        if 'workspace' in cargo_toml:
            cargo_toml['workspace']['package']['version'] = new_version
        else:
            cargo_toml['package']['version'] = new_version

        # Write back to file using custom TOML writer
        write_toml(cargo_toml, path)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error updating {path}: {e}")
        sys.exit(1)

def validate_version(version: str) -> bool:
    """Validate semver format."""
    semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    return re.match(semver_pattern, version) is not None

def parse_version(version: str) -> tuple[int, int, int, str | None]:
    """Parse version into components (major, minor, patch, prerelease)."""
    match = re.match(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(.+))?$', version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch, prerelease = match.groups()
    return int(major), int(minor), int(patch), prerelease

def bump_prerelease(current_version: str) -> str:
    """Bump prerelease version (e.g., beta.1 -> beta.2, rc.1 -> rc.2)."""
    major, minor, patch, prerelease = parse_version(current_version)

    if not prerelease:
        # No prerelease, default to beta.1
        return f"{major}.{minor}.{patch}-beta.1"

    # Parse prerelease (e.g., "beta.1", "rc.2")
    match = re.match(r'^([a-zA-Z]+)\.(\d+)$', prerelease)
    if match:
        label, num = match.groups()
        new_num = int(num) + 1
        return f"{major}.{minor}.{patch}-{label}.{new_num}"
    else:
        # If no number, add .1
        return f"{major}.{minor}.{patch}-{prerelease}.1"

def bump_version(current_version: str, bump_type: str, prerelease_type: str | None = None) -> str:
    """Increment version based on bump type."""
    major, minor, patch, current_prerelease = parse_version(current_version)

    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    elif bump_type == 'prerelease':
        return bump_prerelease(current_version)
    elif bump_type == 'release':
        # Remove prerelease suffix
        return f"{major}.{minor}.{patch}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    # Add prerelease suffix if specified
    base_version = f"{major}.{minor}.{patch}"
    if prerelease_type:
        return f"{base_version}-{prerelease_type}.1"

    # Default to beta.1 for non-release bumps
    return f"{base_version}-beta.1"

def sync_lockfile() -> bool:
    """Refresh Cargo.lock so workspace member versions match Cargo.toml.

    Cargo rewrites the version of every workspace member inside Cargo.lock, so
    staging only Cargo.toml leaves the lockfile stale until the next cargo
    command dirties it again. Uses --offline because the lock is already fully
    resolved; only local crate versions change, so no index access is needed.
    """
    result = subprocess.run(
        ['cargo', 'metadata', '--format-version', '1', '--offline'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print('Warning: could not refresh Cargo.lock; run `cargo metadata` and'
              ' commit the lockfile before tagging')
        return False

    return True

def git_commit_and_tag(version: str) -> None:
    """Commit version change and create git tag."""
    files = ['Cargo.toml']

    if sync_lockfile():
        files.append('Cargo.lock')

    subprocess.run(['git', 'add', *files], check=True)
    subprocess.run(['git', 'commit', '-m', f'Bump version to {version}'], check=True)
    subprocess.run(['git', 'tag', f'v{version}'], check=True)

def main():
    parser = argparse.ArgumentParser(description='Manage project versioning')
    parser.add_argument('bump',
                        choices=['major', 'minor', 'patch', 'prerelease', 'release', 'set'],
                        help='Type of version bump')
    parser.add_argument('version', nargs='?', default=None,
                        help='Specific version to set (only used with "set" bump type)')
    parser.add_argument('--pre', '--prerelease',
                        choices=['beta', 'rc', 'alpha'],
                        help='Prerelease type (beta, rc, alpha) for major/minor/patch bumps')

    args = parser.parse_args()

    cargo_toml_path = 'Cargo.toml'

    # Read current version
    current_version = read_cargo_toml(cargo_toml_path)

    # Determine new version
    if args.bump == 'set':
        if not args.version:
            print("Error: Must provide a version when using 'set'")
            sys.exit(1)

        # Validate provided version
        if not validate_version(args.version):
            print(f"Invalid version format: {args.version}")
            sys.exit(1)

        new_version = args.version
    else:
        new_version = bump_version(current_version, args.bump, args.pre)

    # Update Cargo.toml
    update_cargo_toml(cargo_toml_path, new_version)

    # Commit and tag
    git_commit_and_tag(new_version)

    print(f"Updated version from {current_version} to {new_version}")
    print("Created git commit and tag")

if __name__ == '__main__':
    main()