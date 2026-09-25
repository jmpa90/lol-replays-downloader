import requests
import time
from collections import deque
import json
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

API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {"X-Riot-Token": API_KEY}

MAX_REQUESTS_PER_SECOND = 20
MAX_REQUESTS_PER_2_MIN = 100
TIME_WINDOW_2_MIN = 120
REQUEST_TIMEOUT = 60  # segundos; sin timeout un socket colgado bloquea el run entero

PLAYERS_CSV = "data/players.csv"
INDEX_FILE = "index.json"
EXPIRED_FILE = "expired_replays.json"

# Opcional: solo bajar replays de partidas de los últimos N días.
MAX_AGE_DAYS = os.getenv("REPLAY_MAX_AGE_DAYS")

# account-v1 solo se sirve desde americas/asia/europe: "sea" (válido para
# match-v5) da 403 ahí -- ver TiSaD#TiSaD, 2026-09-09.
ACCOUNT_REGION = {"sea": "asia"}

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

        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)

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
# LOAD PLAYERS / STATE
# =====================
def load_players():
    with open(PLAYERS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        return list(reader)


def load_known_match_ids(path=INDEX_FILE):
    """Match ids ya subidos a Drive (según index.json). El runner de Actions
    arranca siempre con replays/ vacío, así que el viejo chequeo
    `os.path.exists` nunca saltaba nada: cada run re-bajaba y re-subía los
    ~5 replays de cada jugador aunque ya estuvieran en Drive."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return set()
    return {
        os.path.splitext(item["file_name"])[0]
        for item in index
        if item.get("file_name")
    }


def load_expired_replays(path=EXPIRED_FILE):
    """Match ids cuyo replay ya dio 403/404 permanente: no reintentarlos."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def save_expired_replays(ids, path=EXPIRED_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(set(ids)), f, indent=2)

# =====================
# AGE FILTER
# =====================
def _max_age_start_time(days, now=None):
    """Epoch seconds de hace `days` días (para el startTime de match-v5)."""
    if now is None:
        now = time.time()
    return int(now - days * 86400)


def _should_skip(match_id, expired_ids, recent_ids):
    """recent_ids=None significa "edad desconocida, no filtrar por edad"."""
    if match_id in expired_ids:
        return True
    if recent_ids is not None and match_id not in recent_ids:
        return True
    return False


def get_recent_match_ids(puuid, region, days):
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"startTime": _max_age_start_time(days), "count": 100}
    return set(safe_get(url, headers=HEADERS, params=params).json())

# =====================
# GET PUUID
# =====================
def get_puuid(player):
    region = ACCOUNT_REGION.get(player["region"], player["region"])
    url = (
        f"https://{region}.api.riotgames.com"
        f"/riot/account/v1/accounts/by-riot-id/"
        f"{player['riotIdGameName']}/{player['riotIdTagline']}"
    )
    data = safe_get(url, headers=HEADERS).json()
    return data["puuid"]

# =====================
# EXTRACT MATCH ID
# =====================
def extract_match_id(replay_url, metadata=None):
    """Pull the real match id (e.g. "KR_8375243570") out of a Riot replay
    URL, or None if it can't be determined.

    Riot's S3 object key used to BE the match id (".../KR_123.replay").
    Riot has since started serving replays from a generic per-match-folder
    key instead (".../kr_8375243570/0.replay?...&response-content-disposition=
    attachment%3B%20filename%3D%22KR_8375243570.rofl%22&..."), and matching
    only `/([^/]+)\\.replay` returned the literal "0" for every replay -- every
    replay in a region collided on "replays/<region>/0.rofl" (real incident,
    2026-09-09, Hide on bush#KR1).

    Order of preference:
      1. the `response-content-disposition` quoted filename (exact casing),
      2. the old "/{matchId}.replay" key,
      3. the "/{matchId}/0.replay" folder, upper-cased,
      4. metadata["matchId"], if given.
    """
    if replay_url:
        query = parse_qs(urlparse(replay_url).query)
        disposition = query.get("response-content-disposition", [None])[0]
        if disposition:
            m = re.search(r'filename="?([^"&;]+)\.rofl"?', unquote(disposition))
            if m:
                return m.group(1)

        path = urlparse(replay_url).path
        m = re.search(r"/([^/]+)\.replay$", path)
        if m and not m.group(1).isdigit():
            return m.group(1)

        m = re.search(r"/([A-Za-z0-9]+_\d+)/\d+\.replay$", path)
        if m:
            return m.group(1).upper()

    if metadata and metadata.get("matchId"):
        return metadata["matchId"].upper()

    return None


# =====================
# DOWNLOAD REPLAYS
# =====================
def download_file(url, file_path):
    """Descarga a un .part y renombra al final: un corte a mitad de
    descarga no deja un .rofl truncado que upload_replay.py subiría."""
    tmp_path = file_path + ".part"
    # Sin X-Riot-Token: es una URL pre-firmada de un CDN externo, la API key
    # no tiene nada que hacer ahí (y no cuenta contra el rate limit de Riot).
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(tmp_path, file_path)


def download_replays(puuid, region, known_ids, expired_ids):
    replay_folder = f"replays/{region}"
    os.makedirs(replay_folder, exist_ok=True)

    recent_ids = None
    if MAX_AGE_DAYS:
        recent_ids = get_recent_match_ids(puuid, region, int(MAX_AGE_DAYS))

    url = (f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/replays")
    replays = safe_get(url, headers=HEADERS).json().get("matchFileURLs", [])

    for replay_url in replays:
        match_id = extract_match_id(replay_url)
        if match_id is None:
            print(f"⚠️ No se pudo extraer el match id de {replay_url}, se omite")
            continue

        if match_id in known_ids or _should_skip(match_id, expired_ids, recent_ids):
            continue

        file_path = os.path.join(replay_folder, f"{match_id}.rofl")
        if os.path.exists(file_path):
            continue

        try:
            download_file(replay_url, file_path)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (403, 404):
                print(f"🗑️ {match_id} ya no está disponible ({status}), marcado como expirado")
                expired_ids.add(match_id)
                continue
            raise

        known_ids.add(match_id)
        print(f"✅ Guardado {match_id}.rofl ({region})")

# =====================
# MAIN
# =====================
def main():
    if not API_KEY:
        raise RuntimeError("RIOT_API_KEY no está seteada")

    players = load_players()
    known_ids = load_known_match_ids()
    expired_ids = load_expired_replays()
    initial_expired = set(expired_ids)
    print(f"👥 Jugadores cargados: {len(players)}")
    print(f"📚 Replays ya en index.json: {len(known_ids)}, expirados: {len(expired_ids)}")

    for player in players:
        print(
            f"🔎 {player['riotIdGameName']}#{player['riotIdTagline']} "
            f"({player['region']})"
        )
        try:
            puuid = get_puuid(player)
            download_replays(puuid, player["region"], known_ids, expired_ids)
        except Exception as exc:
            # One bad account (renamed/banned Riot ID, region typo, a 403/404
            # from account-v1, ...) used to crash the whole run here and
            # skip every player after it in the CSV. Log and keep going instead.
            print(
                f"⚠️ Error con {player['riotIdGameName']}#{player['riotIdTagline']} "
                f"({player['region']}): {exc}"
            )

    if expired_ids != initial_expired:
        save_expired_replays(expired_ids)

if __name__ == "__main__":
    main()
