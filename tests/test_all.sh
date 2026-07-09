#!/bin/bash
set -e  # Exit immediately if any command fails

for f in tests/test_*.mojo; do
    echo "--- $f ---"
    mojo run -I . -I tests/ "$f"
done
