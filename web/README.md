# Test console (React)

A browser UI for trying the RAG pipeline: ask a question, see the answer with clickable citations,
and inspect exactly which chunks were retrieved and what they scored.

```bash
# terminal 1, repo root, venv active
uvicorn api.main:app --port 8010 --reload

# terminal 2
cd web && npm install && npm run dev      # http://localhost:5180 (proxies /api to :8010)
```

**Search only (free)** runs retrieval alone. **Ask Claude** also generates an answer (~$0.02).

Smoke test (needs both servers running): `npm run smoke` (free) or `npm run smoke:ask` (one real
question). Screenshots go to `e2e/screenshots/`.
