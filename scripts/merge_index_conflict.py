"""Resolve a git rebase conflict on index.json and/or expired_replays.json
by unioning both versions.

Used by .github/workflows/upload_replays.yml's "Commit and push index.json"
step: if pushing our update is rejected because origin/main moved, we
rebase onto it; if that rebase conflicts on either tracked JSON file (both
sides appended different new entries), we don't want to pick either side
exclusively (git checkout --ours/--theirs would silently drop that run's
uploads, or a run's newly-discovered permanent-404 replay ids). Instead we
merge both sides and let the rebase continue with the union.

Run with the repo root as cwd, mid-conflict (conflict markers / stage 2 and
3 blobs available via `git show`):

    python scripts/merge_index_conflict.py

Each file is only touched if it actually has conflicting stage-2/stage-3
blobs; a file that wasn't part of the conflict is left untouched.
"""
import json
import subprocess


def load_side(rev, path):
    try:
        out = subprocess.check_output(
            ["git", "show", f"{rev}:{path}"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        return json.loads(out)
    except Exception:
        return None


def merge_index(theirs, ours):
    """Union two index.json entry lists, keyed by drive_file_id (falling
    back to file_name), later entries (ours) winning on key collision."""
    merged = {}
    for item in theirs + ours:
        key = item.get("drive_file_id") or item.get("file_name")
        merged[key] = item
    return list(merged.values())


def merge_expired(theirs, ours):
    """Union two expired_replays.json match-id lists."""
    return sorted(set(theirs) | set(ours))


def resolve(path, merge_fn):
    ours = load_side(":2", path)    # our new commit's version
    theirs = load_side(":3", path)  # latest origin/main version

    if ours is None and theirs is None:
        return  # this file wasn't part of the conflict, leave it as-is

    merged = merge_fn(theirs or [], ours or [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def main():
    resolve("index.json", merge_index)
    resolve("expired_replays.json", merge_expired)


if __name__ == "__main__":
    main()
