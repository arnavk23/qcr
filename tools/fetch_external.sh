#!/usr/bin/env bash
# This code is part of qcr.
#
# (C) Copyright Arnav Kapoor 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

# Download third-party material that is not redistributed in this repository:
#   external/benchmarks/   VOQC (Nam) and QASMBench circuits, pinned commits   [experiments/gen_bench.py]
#   external/QCOptimDemo/  the authors' MATLAB demo (NJP paper, ref. [38])      [experiments/exp_figures.py demo]
#   external/paper/        arXiv PDF and LaTeX source of arXiv:2502.14715       [reference only]
#
# Usage: tools/fetch_external.sh [benchmarks|demo|paper|all]   (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${QCR_EXTERNAL:-$ROOT/external}"
VOQC_URL=https://github.com/inQWIRE/VOQC-benchmarks.git
VOQC_REV=3582e1b7df5a88215905d2f591307214fe965165
QASMBENCH_URL=https://github.com/pnnl/QASMBench.git
QASMBENCH_REV=357b942396d5c2b7cbc1c229c585a6ef5ccaebac
DEMO_URL=https://www.tnt.uni-hannover.de/en/staff/rosenhahn/QCOptimDemo.zip
ARXIV_ID=2502.14715

clone_at() {  # url rev dir
  if [ -d "$3/.git" ]; then echo "  $3 exists, skipping"; return; fi
  git clone --quiet --filter=blob:none "$1" "$3"
  git -C "$3" checkout --quiet "$2"
  echo "  $3 @ ${2:0:10}"
}

fetch_benchmarks() {
  echo "benchmarks -> $DEST/benchmarks"
  mkdir -p "$DEST/benchmarks"
  clone_at "$VOQC_URL" "$VOQC_REV" "$DEST/benchmarks/VOQC-benchmarks"
  clone_at "$QASMBENCH_URL" "$QASMBENCH_REV" "$DEST/benchmarks/QASMBench"
}

fetch_demo() {
  echo "authors' MATLAB demo -> $DEST/QCOptimDemo"
  if [ -d "$DEST/QCOptimDemo" ]; then echo "  exists, skipping"; return; fi
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  curl -fsSL "$DEMO_URL" -o "$tmp/demo.zip"
  unzip -q "$tmp/demo.zip" -d "$tmp/unz"
  src="$(dirname "$(find "$tmp/unz" -name 'optimCodeGMode1DComp.m' -print -quit)")"
  mkdir -p "$DEST"; mv "$src" "$DEST/QCOptimDemo"
}

fetch_paper() {
  echo "paper -> $DEST/paper"
  mkdir -p "$DEST/paper/src"
  curl -fsSL "https://arxiv.org/pdf/$ARXIV_ID" -o "$DEST/paper/paper.pdf"
  curl -fsSL "https://arxiv.org/e-print/$ARXIV_ID" | tar -xz -C "$DEST/paper/src"
}

case "${1:-all}" in
  benchmarks) fetch_benchmarks ;;
  demo) fetch_demo ;;
  paper) fetch_paper ;;
  all) fetch_benchmarks; fetch_demo; fetch_paper ;;
  *) echo "usage: $0 [benchmarks|demo|paper|all]" >&2; exit 2 ;;
esac
