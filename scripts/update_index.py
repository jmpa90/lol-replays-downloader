import json
import os

INDEX_FILE = "index.json"
TEMP_FILE = "uploaded_temp.json"

# Crear index.json si no existe
if not os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "w") as f:
        json.dump([], f, indent=2)

# Leer index.json actual
with open(INDEX_FILE, "r") as f:
    index = json.load(f)

# Leer metadata de los archivos recién subidos
if os.path.exists(TEMP_FILE):
    with open(TEMP_FILE, "r") as f:
        new_uploads = json.load(f)

    # Agregar solo los que no están en index
    known_ids = {x['drive_file_id'] for x in index}
    for item in new_uploads:
        if item['drive_file_id'] not in known_ids:
            index.append(item)
            known_ids.add(item['drive_file_id'])

    # Guardar index actualizado
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

    # Borrar temporal
    os.remove(TEMP_FILE)
