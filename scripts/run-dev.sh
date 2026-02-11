#!/bin/bash

# Caminho absoluto do diretório do script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Caminho absoluto para a pasta config
CONFIG_DIR="$SCRIPT_DIR/../config"

docker compose \
  -f "$CONFIG_DIR/docker-compose-dev.yml" \
  -f "$CONFIG_DIR/docker-compose.dozzle.yml" \
  up -d
