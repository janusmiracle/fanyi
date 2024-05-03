#!/bin/bash

DATA_PATH="./dugong/data/"

clean_directory() {
	# Find a way to ignore .gitkeep files
	find "$1" -mindepth 1 -delete
}

clean_directory "$DATA_PATH"

echo "Data folder has been cleaned."
