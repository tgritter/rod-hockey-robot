#!/bin/sh -x
cd `dirname $0`

# --collect-all viam bundles the SDK's native libs (libviam_rust_utils), which
# the connect-to-self RobotClient needs at runtime and which PyInstaller's import
# analysis otherwise misses.
uv run -m PyInstaller --onefile --collect-all viam --hidden-import="googleapiclient" module.py

TAR_FILES="meta.json dist/module app/dist"
FIRST_RUN=$(uv run python -c "import json; print(json.load(open('meta.json')).get('first_run', ''))" 2>/dev/null)
if [ -n "$FIRST_RUN" ] && [ -f "$FIRST_RUN" ]; then
    TAR_FILES="$TAR_FILES $FIRST_RUN"
fi
tar -czvf dist/archive.tar.gz $TAR_FILES
