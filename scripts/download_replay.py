import requests
import time
from collections import deque
import os
import re
import csv

# =====================
# CONFIG
# =====================
from dotenv import load_dotenv

# Cargar variables desde el .env
load_dotenv()  # Esto lee .env automáticamente

# Ahora sí puedes usarla
API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise RuntimeError("RIOT_API_KEY no está seteada")

HEADERS = {"X-Riot-Token": API_KEY}

MAX_REQUESTS_PER_SECOND = 20
MAX_REQUESTS_PER_2_MIN = 100
TIME_WINDOW_2_MIN = 120

PLAYERS_CSV = "data/players.csv"

request_times = deque()

# =====================
# RATE-LIMIT SAFE GET
# =====================
def safe_get(url, headers, params=None, max_retries=5):
    global request_times

    for _ in range(max_retries):
        now = time.time()

        # ventana 2 minutos
        while request_times and now - request_times[0] > TIME_WINDOW_2_MIN:
            request_times.popleft()

        if len(request_times) >= MAX_REQUESTS_PER_2_MIN:
            sleep_time = TIME_WINDOW_2_MIN - (now - request_times[0]) + 2
            print(f"⏳ Rate limit global, esperando {sleep_time:.1f}s")
            time.sleep(sleep_time)
            continue

        # burst limit
        if request_times and now - request_times[-1] < 1 / MAX_REQUESTS_PER_SECOND:
            time.sleep(1 / MAX_REQUESTS_PER_SECOND)

        r = requests.get(url, headers=headers, params=params)

        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", 1))
            print(f"⚠️ 429 recibido, esperando {retry_after}s")
            time.sleep(retry_after)
            continue

        r.raise_for_status()
        request_times.append(time.time())
        return r

    raise RuntimeError("Demasiados 429, abortando")

# =====================
# LOAD PLAYERS
# =====================
def load_players():
    with open(PLAYERS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        return list(reader)

# =====================
# ACCOUNT-V1 ROUTING
# =====================
def account_routing(region):
    """account-v1 is not served on sea.api.riotgames.com (returns 403);
    match-v5/replay endpoints ARE served on sea and must stay untouched.
    Mirrors riftcast's account_routing(): sea -> asia for account-v1 only."""
    if region == "sea":
        return "asia"
    return region

# =====================
# GET PUUID
# =====================
def get_puuid(player):
    account_region = account_routing(player["region"])
    url = (
        f"https://{account_region}.api.riotgames.com"
        f"/riot/account/v1/accounts/by-riot-id/"
        # f"{player['riotIdGameName']}/{player['riotIdTagline']}"
        f"{player['riotIdGameName'].replace(' ', '%20')}/{player['riotIdTagline']}"
    )
    data = safe_get(url, headers=HEADERS).json()
    return data["puuid"]

# =====================
# MATCH ID EXTRACTION
# =====================
MATCH_ID_PATTERNS = (
    re.compile(r"/([^/]+)/0\.replay"),
    re.compile(r"/([^/]+)\.replay"),
)


def extract_match_id(replay_url, metadata=None):
    """Extract the match id from a replay download URL.

    Tries the current `/{matchId}/0.replay` shape first, then the older
    `/{matchId}.replay` shape, then falls back to a `matchId` key on an
    optional metadata dict (e.g. a parsed API response). Returns None
    (instead of raising) when nothing matches, so callers can skip the
    offending replay rather than crash the whole player loop.
    """
    if replay_url:
        for pattern in MATCH_ID_PATTERNS:
            match = pattern.search(replay_url)
            if match:
                return match.group(1).upper()

    if metadata and metadata.get("matchId"):
        return str(metadata["matchId"]).upper()

    return None


def _truncate_url(url, max_len=80):
    """Truncate a URL for logging, dropping any signed query string."""
    if not url:
        return "<empty>"
    base = url.split("?", 1)[0]
    if len(base) > max_len:
        return base[:max_len] + "..."
    return base

# =====================
# DOWNLOAD REPLAYS
# =====================
def download_replays(puuid, region):
    replay_folder = f"replays/{region}"
    os.makedirs(replay_folder, exist_ok=True)

    url = (f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/replays")
    replays = safe_get(url, headers=HEADERS).json().get("matchFileURLs", [])

    for replay_url in replays:
        match_id = extract_match_id(replay_url)
        if match_id is None:
            print(f"⚠️ No se pudo extraer match_id de {_truncate_url(replay_url)}, se omite")
            continue

        file_path = os.path.join(replay_folder, f"{match_id}.rofl")

        if os.path.exists(file_path):
            continue

        try:
            r = safe_get(replay_url, headers=HEADERS)

            with open(file_path, "wb") as f:
                f.write(r.content)

            print(f"✅ Guardado {match_id}.rofl ({region})")
        except Exception as e:
            print(f"❌ Error descargando {match_id} ({region}): {e}")
            print("➡️ Continuando con el siguiente replay...\n")
            continue

# =====================
# MAIN
# =====================
def main():
    players = load_players()
    print(f"👥 Jugadores cargados: {len(players)}")

    for player in players:
        try:
            print(
                f"🔎 {player['riotIdGameName']}#{player['riotIdTagline']} "
                f"({player['region']})"
            )

            puuid = get_puuid(player)
            download_replays(puuid, player["region"])

        except Exception as e:
            print(
                f"❌ Error con {player['riotIdGameName']}#{player['riotIdTagline']} "
                f"({player['region']}): {e}"
            )
            print("➡️ Continuando con el siguiente jugador...\n")
            continue


if __name__ == "__main__":
    main()
