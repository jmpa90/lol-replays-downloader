import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

REPLAY_FOLDER = "replays"
DRIVE_FOLDER_ID = "TU_DRIVE_FOLDER_ID"

# Leer el JSON del secret
creds_json = os.environ.get("GOOGLE_DRIVE_CREDENTIALS")
if not creds_json:
    raise ValueError("No se encontró la variable de entorno GOOGLE_DRIVE_CREDENTIALS")

creds = service_account.Credentials.from_service_account_info(
    json.loads(creds_json),
    scopes=["https://www.googleapis.com/auth/drive.file"]
)

service = build("drive", "v3", credentials=creds)

# =====================
# FUNCIONES AUXILIARES
# =====================
def find_file_in_drive(name, folder_id):
    """Busca un archivo por nombre dentro de una carpeta de Drive"""
    query = f"'{folder_id}' in parents and name='{name}' and trashed=false"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    files = results.get('files', [])
    if files:
        return files[0]  # devuelve el primer match
    return None

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
                file_metadata = {
                    "name": file_name,
                    "parents": [DRIVE_FOLDER_ID]
                }
                media = MediaFileUpload(local_path, resumable=True)

                # Revisar si ya existe en Drive
                existing_file = find_file_in_drive(file_name, DRIVE_FOLDER_ID)
                if existing_file:
                    # Si existe, actualizar contenido
                    uploaded_file = service.files().update(
                        fileId=existing_file['id'],
                        media_body=media,
                        fields='id,name,webViewLink'
                    ).execute()
                    action = "Actualizado"
                else:
                    # Si no existe, crear
                    uploaded_file = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id,name,webViewLink'
                    ).execute()
                    action = "Subido"

                print(f"{action}: {uploaded_file['name']}")
                print(f"ID: {uploaded_file['id']}")
                print(f"URL: {uploaded_file['webViewLink']}")

                uploaded_metadata.append({
                    "file_name": uploaded_file['name'],
                    "drive_file_id": uploaded_file['id'],
                    "webViewLink": uploaded_file['webViewLink'],
                    "local_path": local_path
                })

                # Borrar archivo local con retry en Windows
                for attempt in range(5):
                    try:
                        os.remove(local_path)
                        print(f"{file_name} eliminado localmente.\n")
                        break
                    except PermissionError:
                        time.sleep(0.5)

            except Exception as e:
                print(f"Error subiendo {local_path}: {e}")

# =====================
# GUARDAR METADATA
# =====================
if uploaded_metadata:
    with open("uploaded_temp.json", "w") as f:
        json.dump(uploaded_metadata, f, indent=2)
    print(f"Metadata de {len(uploaded_metadata)} archivos guardada en uploaded_temp.json")
else:
    print("No se encontraron archivos .rofl para subir.")
