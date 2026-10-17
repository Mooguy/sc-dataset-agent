# sc-dataset-agent

Plain-language request → ranked public scRNA-seq datasets (GEO, CELLxGENE, Ensembl) with metadata and a download plan.

Status: Day 0 (setup).

## Local setup

```bash
uv sync
cp .env.example .env        # fill in keys
docker compose up -d        # local Postgres
uv run scripts/test_models.py --list   # see available model IDs
uv run scripts/test_models.py          # compare tool calling across CANDIDATE_MODELS
```
