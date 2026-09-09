import requests
import time
from collections import deque
import os
import re
import csv
from urllib.parse import urlparse, parse_qs, unquote

# =====================
# CONFIG
# =====================
from dotenv import load_dotenv

# Cargar variables desde el .env
load_dotenv()  # Esto lee .env automáticamente

# Ahora sí puedes usarla
API_KEY = os.getenv("RIOT_API_KEY")
print(API_KEY)

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
# GET PUUID
# =====================
def get_puuid(player):
    url = (
        f"https://{player['region']}.api.riotgames.com"
        f"/riot/account/v1/accounts/by-riot-id/"
        f"{player['riotIdGameName']}/{player['riotIdTagline']}"
    )
    data = safe_get(url, headers=HEADERS).json()
    return data["puuid"]

# =====================
# EXTRACT MATCH ID
# =====================
def extract_match_id(replay_url):
    """Pull the real match id (e.g. "KR_8375243570") out of a Riot replay
    URL. Riot's S3 object key used to BE the match id (".../KR_123.replay"),
    which is what the old `/([^/]+)\\.replay` regex assumed. Riot has since
    started serving some/all replays from a generic per-match-folder key
    instead (".../kr_8375243570/0.replay?...&response-content-disposition=
    attachment%3B%20filename%3D%22KR_8375243570.rofl%22&..."), so that regex
    now matches the literal "0" for every replay -- every replay in a region
    collided on the same "replays/<region>/0.rofl" path, and after the
    first save `os.path.exists` silently skipped every replay after it.
    (Real incident, 2026-09-09: this exact pattern to Hide on bush#KR1's
    replays only ever downloaded one "0.rofl" per region.)

    Prefer the `response-content-disposition`'s quoted filename (present on
    every URL seen so far, and it's the one place Riot gives the *exact*,
    correctly-cased match id) -- fall back to the old path-segment regex
    for any URL shape that doesn't have it, so this never regresses the
    working case.
    """
    query = parse_qs(urlparse(replay_url).query)
    disposition = query.get("response-content-disposition", [None])[0]
    if disposition:
        m = re.search(r'filename="?([^"&;]+)\.rofl"?', unquote(disposition))
        if m:
            return m.group(1)

    m = re.search(r"/([^/]+)\.replay", replay_url)
    if m and m.group(1) != "0":
        return m.group(1)

    raise ValueError(f"Could not extract match id from replay URL: {replay_url}")


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
        file_path = os.path.join(replay_folder, f"{match_id}.rofl")

        if os.path.exists(file_path):
            continue

        r = safe_get(replay_url, headers=HEADERS)

        with open(file_path, "wb") as f:
            f.write(r.content)

        print(f"✅ Guardado {match_id}.rofl ({region})")

# =====================
# MAIN
# =====================
def main():
    players = load_players()
    print(f"👥 Jugadores cargados: {len(players)}")

    for player in players:
        print(
            f"🔎 {player['riotIdGameName']}#{player['riotIdTagline']} "
            f"({player['region']})"
        )
        try:
            puuid = get_puuid(player)
            download_replays(puuid, player["region"])
        except Exception as exc:
            # One bad account (renamed/banned Riot ID, region typo, a 403/404
            # from account-v1, ...) used to crash the whole run here and
            # skip every player after it in the CSV -- see TiSaD#TiSaD
            # (region "sea") 403'ing and taking the rest of the roster down
            # with it, 2026-09-09. Log and keep going instead.
            print(
                f"⚠️ Error con {player['riotIdGameName']}#{player['riotIdTagline']} "
                f"({player['region']}): {exc}"
            )

if __name__ == "__main__":
    main()
