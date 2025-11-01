#!/bin/bash
set -euo pipefail

# EUOS25 Validation Scores Checker
# This script checks cross-validation scores from trained models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "==================================="
echo "EUOS25 Validation Scores Check"
echo "==================================="
echo ""

uv run python3 "$SCRIPT_DIR/check_scores.py"

echo ""
echo "==================================="
echo "Check completed"
echo "==================================="

