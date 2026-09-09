"""Regenerate data/players.csv from riftcast's own roster
(K:\\Dev\\projects\\riftcast\\config\\players.csv), so every roster row --
including new creator-watch signings -- is covered by this downloader
without hand-editing two CSVs in two repos.

Why this exists: data/players.csv here used to be maintained by hand,
copy-pasted (partially, and inconsistently) from riftcast's CSV each time
the roster changed. That's how two 2026-09-07 signings (naruto#helix and
미소녀#mio韩服, both "Yasuo Mid KR" -- see riftcast's
data/state/creator_watch.json, video Jv5hB3V_4H8) went untracked here for
2 days: nobody re-ran the copy-paste. Riot's match/v5/matches/by-puuid/
{puuid}/replays endpoint only ever returns the last ~5 replays per
account, so any account missing from this list for more than a
handful of games has *already* lost the replays for the games it
missed -- there is no backfill once a replay rotates out of that window
(see riftcast's scripts/fetch_drive_rofl.py ReplayNotFoundError
docstring for the KR_8355228084 incident). Keeping this file in sync
promptly, ideally every time riftcast's roster changes, is the only way
to not lose coverage.

This only reads riftcast's CSV, never writes it (riftcast is a separate,
independently-owned repo -- this script must never touch it).

IMPORTANT -- this script only works on a machine that has both repos
checked out side by side (riftcast's CSV path is a local filesystem
path, not something reachable from GitHub Actions). The GitHub Actions
workflow (.github/workflows/upload_replays.yml) that actually runs the
downloader on a schedule has NO access to riftcast's repo at all -- it
only ever sees whatever data/players.csv was last committed and pushed
to *this* repo. So the real workflow is:

    1. riftcast's config/players.csv changes (new signing, roster edit)
    2. someone runs this script locally:
           python scripts/sync_players_from_riftcast.py
    3. `git add data/players.csv && git commit && git push`
    4. only THEN does the next scheduled/dispatched Actions run see the
       new accounts.

Step 2-3 are not automated (no local cron/Task Scheduler runs this repo
today -- see README/task notes) -- they're a manual step someone (Claude
or JP) needs to remember to do after riftcast's roster changes. A
tighter fix would be a local scheduled task that runs this + a git push
on a timer, but that's out of scope for this change; flagged for the
"propose a schedule" step of whatever task led you here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Windows consoles are frequently cp1252, which can't encode Korean/Chinese
# riot IDs in this roster -- reconfigure stdout/stderr to UTF-8 (with
# replacement so an odd unmappable byte still doesn't crash the run) rather
# than let a print() of a Korean gameName blow up an otherwise-successful
# sync.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADER_CSV = REPO_ROOT / "data" / "players.csv"

# Local-machine-only path to riftcast's roster. riftcast is READ-ONLY from
# here -- this script must never write to anything under this path.
RIFTCAST_PLAYERS_CSV = Path(r"K:\Dev\projects\riftcast\config\players.csv")

# riftcast's CSV has many more columns (nickName, mainChampion, rank info,
# thumbnail/style stuff for video production, ...) that this downloader has
# no use for -- it only ever needs enough to hit Riot's account-v1 and
# match-v5-replays endpoints.
DOWNLOADER_FIELDS = ["riotIdGameName", "riotIdTagline", "region"]


def load_riftcast_players(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        return list(reader)


def to_downloader_rows(riftcast_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for r in riftcast_rows:
        game_name = r["gameName"].strip()
        tag_line = r["tagLine"].strip()
        region = r["region"].strip()
        if not game_name or not tag_line or not region:
            continue
        rows.append(
            {
                "riotIdGameName": game_name,
                "riotIdTagline": tag_line,
                "region": region,
            }
        )
    return rows


def main() -> int:
    if not RIFTCAST_PLAYERS_CSV.is_file():
        print(
            f"riftcast roster not found at {RIFTCAST_PLAYERS_CSV} -- this script "
            "only works on a machine with riftcast checked out at that path.",
            file=sys.stderr,
        )
        return 1

    riftcast_rows = load_riftcast_players(RIFTCAST_PLAYERS_CSV)
    new_rows = to_downloader_rows(riftcast_rows)

    if DOWNLOADER_CSV.is_file():
        old_rows = load_riftcast_players(DOWNLOADER_CSV)
        old_keys = {(r["riotIdGameName"], r["riotIdTagline"]) for r in old_rows}
    else:
        old_keys = set()
    new_keys = {(r["riotIdGameName"], r["riotIdTagline"]) for r in new_rows}

    added = new_keys - old_keys
    removed = old_keys - new_keys

    with DOWNLOADER_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DOWNLOADER_FIELDS)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"Wrote {len(new_rows)} players to {DOWNLOADER_CSV}")
    if added:
        print(f"  + {len(added)} added:")
        for name, tag in sorted(added):
            print(f"      {name}#{tag}")
    if removed:
        print(f"  - {len(removed)} removed:")
        for name, tag in sorted(removed):
            print(f"      {name}#{tag}")
    if not added and not removed:
        print("  (no changes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
