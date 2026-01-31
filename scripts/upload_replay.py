import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# =====================
# CONFIG
# =====================
REPLAY_FOLDER = "replays"
DRIVE_FOLDER_ID = "1LnxIj6pEmXkib9TogmbtjkERhbLc9b5u"
SCOPES = ["https://www.googleapis.com/auth/drive"]

# =====================
# CARGAR TOKEN DESDE SECRETS
# =====================
token_json = os.environ.get("GOOGLE_DRIVE_TOKEN")
if not token_json:
    raise ValueError("No se encontró la variable de entorno GOOGLE_DRIVE_TOKEN")

creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
service = build("drive", "v3", credentials=creds)

import os
import json
# CAMBIO 1: Importar service_account
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# # =====================
# # CONFIG
# # =====================
# REPLAY_FOLDER = "replays"
# DRIVE_FOLDER_ID = "1LnxIj6pEmXkib9TogmbtjkERhbLc9b5u"
# SCOPES = ["https://www.googleapis.com/auth/drive"]

# # =====================
# # CARGAR TOKEN DESDE SECRETS (SERVICE ACCOUNT)
# # =====================
# # CAMBIO 2: Leer el JSON de la Service Account desde la variable de entorno
# sa_json = os.environ.get("GCP_SERVICE_ACCOUNT") # Antes era GOOGLE_DRIVE_TOKEN
# if not sa_json:
#     raise ValueError("No se encontró la variable de entorno GCP_SERVICE_ACCOUNT")

# # CAMBIO 3: Cargar credenciales usando from_service_account_info
# try:
#     sa_info = json.loads(sa_json)
#     creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
#     service = build("drive", "v3", credentials=creds)
# except Exception as e:
#     raise ValueError(f"Error cargando credenciales de Service Account: {e}")

# =====================
# SUBIDA DE REPLAYS
# =====================
uploaded_metadata = []

print(f"Buscando archivos .rofl en {REPLAY_FOLDER} y sus subcarpetas...")

for root, dirs, files in os.walk(REPLAY_FOLDER):
    for file_name in files:
        if file_name.endswith(".rofl"):
            local_path = os.path.join(root, file_name)
            print(f"Procesando {local_path}...")

            try:
                # Primero buscar si el archivo ya existe en Drive
                query = f"name='{file_name}' and '{DRIVE_FOLDER_ID}' in parents and trashed=false"
                results = service.files().list(q=query, fields="files(id, name)").execute()
                files_in_drive = results.get("files", [])

                if files_in_drive:
                    # Si existe, actualizarlo
                    file_id = files_in_drive[0]["id"]
                    media = MediaFileUpload(local_path, resumable=False)
                    uploaded_file = service.files().update(
                        fileId=file_id,
                        media_body=media,
                        fields="id,name,webViewLink"
                    ).execute()
                    print(f"Actualizado: {uploaded_file['name']} (ID: {uploaded_file['id']})")
                else:
                    # Si no existe, crear nuevo
                    file_metadata = {"name": file_name, "parents": [DRIVE_FOLDER_ID]}
                    media = MediaFileUpload(local_path, resumable=False)
                    uploaded_file = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields="id,name,webViewLink"
                    ).execute()
                    print(f"Subido: {uploaded_file['name']} (ID: {uploaded_file['id']})")

                uploaded_metadata.append({
                    "file_name": uploaded_file["name"],
                    "drive_file_id": uploaded_file["id"],
                    "webViewLink": uploaded_file["webViewLink"],
                    "local_path": local_path
                })

                # Borrar archivo local después de subirlo
                os.remove(local_path)
                print(f"{file_name} eliminado localmente.\n")

            except Exception as e:
                print(f"Error subiendo {local_path}: {e}")

# Guardar metadata temporal
if uploaded_metadata:
    with open("uploaded_temp.json", "w") as f:
        json.dump(uploaded_metadata, f, indent=2)
    print(f"Metadata de {len(uploaded_metadata)} archivos guardada en uploaded_temp.json")
else:
    print("No se encontraron archivos .rofl para subir.")
