#!/bin/bash

DATA_PATH="./dugong/data/"
MODELS_PATH="./dugong/models/"

# Clean the raws and translations directory
clean_directory() {
	# Find a way to ignore .gitkeep files
	find "$1" -mindepth 1 -delete
}

clean_directory "$DATA_PATH"

echo "Data folder has been cleaned."

clean_directory "$MODELS_PATH"

echo "Models folder has been cleaned."
