#!/bin/bash
set -euo pipefail

git pull origin master

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

sudo systemctl restart riego.service

echo "Despliegue completado exitosamente."
