# Sales Lead Qualification Agent

An inbound sales assistant for event venues and marine bookings. It ranks prospects for follow-up, explains why a lead needs review, and keeps a cost and latency record for each batch. The repo uses synthetic examples; it does not contact prospects or pull customer records.

## How it handles a sales batch

1. **Qualify.** LangGraph applies market and budget rules to clear requests. If a lead is ambiguous, a provider extracts a booking signal and a supporting quote.
2. **Route.** The extraction step ranks configured Anthropic, Gemini, and OpenAI models by estimated request cost. It checks the output schema and verifies the quote against the original inquiry. On a timeout, invalid output, or unverifiable quote, it tries the next model. A maximum of three attempts and a shared call quota bound the work.
3. **Review.** The queue lists leads ready for follow-up first, then leads needing review, then out-of-scope requests. Each row includes the decision, supporting evidence when available, and a suggested next step. The agent cannot send outreach or place holds.
4. **Measure.** A sales session reports calls by provider, known token usage, estimated cost, and elapsed time. Attempts with missing provider usage are counted separately. Estimates use configured rates and are not invoice totals. Requests with cached results do not re-count earlier provider calls.

## Try it without an API key

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AGENT_MODE=fixture uvicorn app:app --host 127.0.0.1 --port 8001
```

In another terminal:

```sh
curl -sS http://127.0.0.1:8001/sales/sessions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/sales-session.json
```

See the interactive API docs at http://127.0.0.1:8001/docs. `POST /qualify` handles one lead; `POST /plan` ranks eligible venue quotes from a supplied inventory snapshot. The [Event Inventory & Booking API](https://github.com/Manojh23/event-inventory-booking-api) can supply that snapshot.

## Enable model routing

Set `AGENT_MODE=routed` and provide `MODEL_ROUTES_JSON`, an array of provider, model ID, and input/output USD-per-million-token rates. Choose supported model IDs and set the rates for your account. Only configure providers authorized to handle the inquiry data.

Example shape (replace all placeholder model IDs and illustrative rates):

```sh
export AGENT_MODE=routed
export MODEL_ROUTES_JSON='[
  {"provider":"anthropic","model":"YOUR_ANTHROPIC_MODEL","input_per_million":3,"output_per_million":12},
  {"provider":"gemini","model":"YOUR_GEMINI_MODEL","input_per_million":0.3,"output_per_million":1.2},
  {"provider":"openai","model":"YOUR_OPENAI_MODEL","input_per_million":0.1,"output_per_million":0.4}
]'
```

Supply credentials as environment variables: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (Gemini Developer API), and `OPENAI_API_KEY` for the providers you configure. Never commit keys. `MODEL_CALL_CAP` bounds provider calls per service process (100 by default). The short-lived cache and quota are process-local; the session report is returned to the caller.

## Check regressions

```sh
AGENT_MODE=fixture python -m unittest discover -v
python evaluate.py
```

Tests run without provider calls. They cover cheaper-first routing, fallback through all three providers, rejection of fabricated quotes, shared quotas, lead ordering, cache behavior, and partial-cost reporting. The existing offline benchmark compares rules and caching with an always-call baseline; it does not measure live provider cost, latency, or quality.

After configuring providers, use the paired comparison below to run the same synthetic ambiguous inquiries through a single Anthropic model and the multi-provider router. Both arms bypass the cache. It makes billable calls only when explicitly enabled, and withholds the cost ratio if either arm misses an expected outcome or lacks cost data. Keep the JSON from each run to compare later revisions.

```sh
python compare_routing.py --live --baseline-model YOUR_ANTHROPIC_MODEL
```
