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
# DOWNLOAD REPLAYS
# =====================
def download_replays(puuid, region):
    replay_folder = f"replays/{region}"
    os.makedirs(replay_folder, exist_ok=True)

    url = (f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/replays")
    replays = safe_get(url, headers=HEADERS).json().get("matchFileURLs", [])

    for replay_url in replays:
        match_id = re.search(r"/([^/]+)\.replay", replay_url).group(1)
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
        puuid = get_puuid(player)
        download_replays(puuid, player["region"])

if __name__ == "__main__":
    main()
