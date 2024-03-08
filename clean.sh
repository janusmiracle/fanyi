#!/bin/bash

RAW_PATH="./fanyi/raws/"
TRANSLATIONS_PATH="./fanyi/translations/"

# Clean the raws and translations directory
clean_directory() {
	find "$1" -mindepth 1 -delete
}

clean_directory "$RAW_PATH"
clean_directory "$TRANSLATIONS_PATH"

echo "Raws and translations folders cleaned."
