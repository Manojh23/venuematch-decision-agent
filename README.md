# VenueMatch Decision Agent

A FastAPI service that qualifies an event inquiry, filters a supplied inventory snapshot, and returns ranked quotes or a clear reason for human review. LangGraph coordinates the workflow; four typed LangChain tools handle deterministic checks.

| Tool | Purpose |
| --- | --- |
| `structured_fit` | Check market and minimum budget for the event type. |
| `verify_quote` | Verify that evidence extracted by Claude appears verbatim in the inquiry. |
| `filter_inventory` | Apply capacity, amenities, market, event-type, and date constraints. |
| `rank_quotes` | Prefer options within budget, then closer dates and lower prices. |

Rules handle clear cases. Ambiguous inquiries can take the optional Claude extraction path; its evidence must pass verification. The proposal graph carries eligible venues, quote rankings, and blockers to human review. TTL caching, a model-call quota, and token estimates help control and explain model use.

## Run the service

Use Python 3.12 and start with the deterministic fixture provider:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AGENT_MODE=fixture python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/docs for the request schemas, or try an example:

```sh
curl -sS http://127.0.0.1:8001/qualify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/inquiry.json
```

To enable the Claude adapter, set `AGENT_MODE=claude`, `CLAUDE_MODEL`, and `ANTHROPIC_API_KEY` in your shell. The service accepts an inventory snapshot at `POST /plan`. It can run independently; [Event Inventory & Booking API](https://github.com/Manojh23/event-inventory-booking-api) supplies database-backed inventory when connected.

```sh
python -m unittest discover -v
python evaluate.py
```
