#!/bin/bash
auto-judge run \
    --workflow judges/slm_judges/workflow.crossencode.yml \
    --rag-responses /c/dev/trec/2026-rag/runs/generation \
    --rag-topics /c/dev/trec/2026-rag/topics/trec_rag_2026_queries.jsonl \
    --out-dir ./output-crossencode-rag/
