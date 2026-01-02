#/bin/bash

TEXT_VALUE="$1"  # Capture the text value from the script's command-line argument
BASE_DIR="${PWD##*/}"

docker run --rm \
	-v $(pwd)/src:/app \
	--network host --env-file .env $BASE_DIR python /app/main.py "$TEXT_VALUE"