import requests
import time
from collections import Counter, deque
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

# Per-run counters for the match-v5 /replays endpoint. Riot's patch 26.20
# (2026-10-06) says third-party apps "lose access to replay downloads
# entirely" without naming this endpoint, so we log what it actually returns
# and fail loudly if it stops serving us (see report_endpoint_health).
replays_endpoint_statuses = Counter()   # HTTP status -> count of calls
replays_listed = 0                      # replay URLs returned across players
replays_saved = 0                       # files actually written this run
# Statuses that mean "this endpoint no longer works for us" (as opposed to a
# one-off bad account, which also 403/404s on account-v1 -- see main()).
ENDPOINT_GONE_STATUSES = {401, 403, 404, 410}

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

    global replays_listed, replays_saved

    url = (f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/replays")
    try:
        resp = safe_get(url, headers=HEADERS)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "error"
        replays_endpoint_statuses[status] += 1
        print(f"⚠️ /replays ({region}) respondió HTTP {status}")
        raise
    replays_endpoint_statuses[resp.status_code] += 1
    replays = resp.json().get("matchFileURLs", [])
    replays_listed += len(replays)

    for replay_url in replays:
        match_id = extract_match_id(replay_url)
        file_path = os.path.join(replay_folder, f"{match_id}.rofl")

        if os.path.exists(file_path):
            continue

        r = safe_get(replay_url, headers=HEADERS)

        with open(file_path, "wb") as f:
            f.write(r.content)

        replays_saved += 1
        print(f"✅ Guardado {match_id}.rofl ({region})")

# =====================
# ENDPOINT HEALTH
# =====================
def report_endpoint_health():
    """Print a one-line summary of the /replays endpoint and return False if
    it looks permanently closed to us (every call got 401/403/404/410), so
    main() can fail the step instead of reporting a green "nothing new" run."""
    total = sum(replays_endpoint_statuses.values())
    summary = ", ".join(f"HTTP {k}: {v}" for k, v in sorted(
        replays_endpoint_statuses.items(), key=lambda kv: str(kv[0]))) or "sin llamadas"
    print(f"📊 /replays -> {summary} | listados: {replays_listed} | guardados: {replays_saved}")

    if total == 0:
        return True

    gone = sum(v for k, v in replays_endpoint_statuses.items()
               if k in ENDPOINT_GONE_STATUSES)
    if gone == total:
        print(
            "::error title=Riot /replays endpoint cerrado::"
            f"Todas las llamadas ({total}) a match-v5 /replays dieron "
            f"{sorted(replays_endpoint_statuses)}. Posible efecto del parche 26.20 "
            "(terceros pierden descarga de replays)."
        )
        return False

    if replays_listed == 0 and replays_endpoint_statuses.get(200, 0) == total:
        print(
            "::warning title=Riot /replays sin replays::"
            f"{total} llamadas OK pero 0 replays listados."
        )
    return True


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

    if not report_endpoint_health():
        raise SystemExit(1)

if __name__ == "__main__":
    main()
