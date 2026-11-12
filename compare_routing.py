"""Paired comparison of single-provider and routed extraction on synthetic leads."""
import argparse
import json
import os
from statistics import median
from uuid import NAMESPACE_URL, uuid5

from agent import Lead, Qualifier
from evaluate import CASES
from routing import RoutedModel, load_specs


def compare(specs, baseline_model, router_factory=RoutedModel):
    baseline = next((s for s in specs if s.provider == "anthropic" and s.model == baseline_model), None)
    if baseline is None:
        raise ValueError("The baseline must be an Anthropic model in MODEL_ROUTES_JSON.")
    cases = [
        (Lead(id=uuid5(NAMESPACE_URL, f"paired-sales-{i}"), segment=segment,
              region=region, budget=budget, note=note), decision, quote)
        for i, (segment, region, budget, note, decision, route, quote) in enumerate(CASES)
        if route == "model"
    ]
    def measure(routes):
        agent = Qualifier(router_factory(routes), call_cap=3 * len(cases), ttl=0)
        rows = [(agent.run(lead), decision, quote) for lead, decision, quote in cases]
        available = [row["estimated_usd"] for row, _, _ in rows if row["estimated_usd"] is not None]
        return {
            "cases": len(rows),
            "passed": sum(row["route"] == "model" and row["decision"] == decision
                          and bool(row["quote"]) == quote for row, decision, quote in rows),
            "provider_calls": sum(row["attempted"] for row, _, _ in rows),
            "estimated_cost_usd": round(sum(available), 10) if len(available) == len(rows) else None,
            "median_latency_ms": round(median(row["latency_ms"] for row, _, _ in rows), 3),
        }
    single, routed = measure([baseline]), measure(specs)
    comparable = all(x["passed"] == x["cases"] and x["estimated_cost_usd"] is not None
                     for x in (single, routed))
    return {"dataset": "synthetic ambiguous sales inquiries", "single_provider": single,
            "multi_provider": routed,
            "estimated_cost_ratio": round(single["estimated_cost_usd"] / routed["estimated_cost_usd"], 3)
            if comparable and routed["estimated_cost_usd"] else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable billable API requests for the synthetic test cases.")
    parser.add_argument("--baseline-model", required=True, help="Configured Anthropic model ID.")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required to make provider calls; for offline checks use unittest.")
    print(json.dumps(compare(load_specs(os.environ["MODEL_ROUTES_JSON"]), args.baseline_model), indent=2))
