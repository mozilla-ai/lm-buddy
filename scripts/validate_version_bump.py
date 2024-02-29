import sys

from packaging.version import Version


def validate_version_bump(old_version: str, new_version: str) -> int:
    old_version = Version(old_version)
    new_version = Version(new_version)

    if new_version.major != old_version.major:
        number = "Major"
    elif new_version.minor != old_version.minor:
        number = "Minor"
    elif new_version.micro != old_version.micro:
        number = "Patch"
    else:
        number = "Other"

    if new_version > old_version:
        print(f"Pass - {number} version bump")
        return 0
    elif new_version < old_version:
        print(f"Error - {number} version decreased! {new_version} < {old_version}")
        return 1
    else:
        print(f"Error - version unchanged! {new_version} == {old_version}")
        return 1


if __name__ == "__main__":
    old_version, new_version = sys.argv[1:]
    exit_code = validate_version_bump(old_version, new_version)
    sys.exit(exit_code)
