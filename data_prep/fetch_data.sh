#!/usr/bin/env bash
# Descarga el dataset (imagenes NIR + anotaciones) desde el repo de datos PRIVADO.
#
# Requiere:
#   1) Acceso al repo privado iredondosan/seal-inspection-data (pide invitacion).
#   2) Un token con acceso a ese repo:  export GH_TOKEN=<PAT>
#
# Uso:   export GH_TOKEN=ghp_xxx && bash data_prep/fetch_data.sh
set -euo pipefail
OWNER=iredondosan
DATA_REPO=seal-inspection-data
TAG=v1.0
: "${GH_TOKEN:?Define GH_TOKEN con un PAT con acceso a $OWNER/$DATA_REPO}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
AUTH=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")

echo "consultando release $TAG de $OWNER/$DATA_REPO ..."
mkdir -p .data_dl
curl -fsSL "${AUTH[@]}" "https://api.github.com/repos/$OWNER/$DATA_REPO/releases/tags/$TAG" \
  | python3 -c 'import sys,json
d=json.load(sys.stdin)
for a in d["assets"]: print(a["id"], a["name"])' > .data_dl/_assets.txt

[ -s .data_dl/_assets.txt ] || { echo "No hay assets (¿token sin acceso?)"; exit 1; }

while read -r id name; do
  echo "descargando $name ..."
  curl -fL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" \
    "https://api.github.com/repos/$OWNER/$DATA_REPO/releases/assets/$id" -o ".data_dl/$name"
done < .data_dl/_assets.txt

echo "reconstruyendo data/ ..."
cat .data_dl/seal-data.tar.gz.* | tar xzf -
# hold-out congelado (asset pequeno aparte): holdout_labels.csv, holdout.txt, folds
[ -f .data_dl/seal-holdout.tar.gz ] && tar xzf .data_dl/seal-holdout.tar.gz
rm -rf .data_dl
echo "OK -> data/images + data/annotations + hold-out congelado"
