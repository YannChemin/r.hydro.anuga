# Helpers sourced by the numbered scripts.

set -euo pipefail

# shellcheck source=config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

# Re-run the calling script inside the plumergat GRASS session unless we
# are already in one.
in_grass() {
    if [ -z "${GISBASE:-}" ]; then
        exec grass "$GISDBASE/$PROJECT/$MAPSET" --exec bash "$0" "$@"
    fi
}

# Run an addon: installed command if available, otherwise straight from
# its source tree in $ADDON_SRC/<name>/<name>.py.
run_addon() {
    local name="$1"
    shift
    if command -v "$name" >/dev/null 2>&1; then
        "$name" "$@"
    elif [ -f "$ADDON_SRC/$name/$name.py" ]; then
        python3 "$ADDON_SRC/$name/$name.py" "$@"
    else
        echo "ERROR: addon $name is neither installed nor in $ADDON_SRC/$name" >&2
        exit 1
    fi
}

log() {
    echo "[$(date +%H:%M:%S)] $*"
}
