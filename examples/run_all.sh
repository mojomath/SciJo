#!/bin/bash
# Runs every example end to end. The examples exercise the public API only, so
# a failure here means a user-visible break even when the test suite is green.
#
# Mirrors tests/test_all.sh's invocation: `scijo.mojoc` must have been
# precompiled into tests/ first (`pixi run package`), and both `.` and
# `tests/` go on the import path so the precompiled package resolves.
set -e

if [[ ! -f tests/scijo.mojoc ]]; then
    echo "scijo has not been packaged. Run 'pixi run package' first." >&2
    exit 1
fi

for f in examples/*.mojo; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    mojo run -I . -I tests/ "$f"
done

echo ""
echo "=========================================="
echo "All examples ran successfully!"
echo "=========================================="
