"""Resolve a git rebase conflict on index.json by unioning both versions.

Used by .github/workflows/upload_replays.yml's "Commit and push index.json"
step: if pushing our index.json update is rejected because origin/main moved,
we rebase onto it; if that rebase conflicts on index.json (both sides
appended different newly-uploaded entries), we don't want to pick either
side exclusively (git checkout --ours/--theirs would silently drop that
run's uploads). Instead we merge both sides by drive_file_id and let the
rebase continue with the union.

Run with the repo root as cwd, mid-conflict (index.json has conflict
markers / stage 2 and 3 blobs available via `git show`):

    python scripts/merge_index_conflict.py
"""
import json
import subprocess


def load_side(rev):
    try:
        out = subprocess.check_output(["git", "show", rev]).decode("utf-8")
        return json.loads(out)
    except Exception:
        return []


def merge(theirs, ours):
    """Union two index.json entry lists, keyed by drive_file_id (falling
    back to file_name), later entries (ours) winning on key collision."""
    merged = {}
    for item in theirs + ours:
        key = item.get("drive_file_id") or item.get("file_name")
        merged[key] = item
    return list(merged.values())


def main():
    ours = load_side(":2:index.json")    # our new commit's version
    theirs = load_side(":3:index.json")  # latest origin/main version

    with open("index.json", "w") as f:
        json.dump(merge(theirs, ours), f, indent=2)


if __name__ == "__main__":
    main()
