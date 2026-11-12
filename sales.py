"""Review queue and batch-level usage reporting for inbound sales leads."""
from collections import Counter
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent import Lead


class SalesSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leads: list[Lead] = Field(min_length=1, max_length=20)

    @field_validator("leads")
    @classmethod
    def distinct_leads(cls, leads):
        if len({lead.id for lead in leads}) != len(leads):
            raise ValueError("Each lead ID must be unique within a sales session.")
        return leads


def build_sales_queue(request, qualifier):
    start = perf_counter()
    rows = []
    for lead in request.leads:
        outcome = qualifier.run(lead)
        decision = outcome["decision"]
        action = (
            "Follow up to confirm the event date and requirements." if decision == "shortlist"
            else "Ask the prospect to confirm their budget before proposing a venue."
            if outcome["quote"]
            else "Review the inquiry before contacting the prospect." if decision == "review"
            else "No outreach under the current sales criteria."
        )
        rows.append({"lead_id": lead.id, "segment": lead.segment, "region": lead.region,
                     "next_action": action, **outcome})
    order = {"shortlist": 0, "review": 1, "out_of_scope": 2}
    rows.sort(key=lambda x: (order[x["decision"]], not bool(x["quote"])))
    attempts = [a for row in rows for a in row["provider_attempts"]]
    priced = [a for a in attempts if a["estimated_usd"] is not None]
    by_provider = Counter(a["provider"] for a in attempts)
    return {
        "session_id": uuid4(),
        "leads": rows,
        "metrics": {
            "decisions": dict(Counter(x["decision"] for x in rows)),
            "cache_hits": sum(x["route"] == "cache" for x in rows),
            "model_calls": sum(x["attempted"] for x in rows),
            "calls_by_provider": dict(by_provider),
            "reported_input_tokens": sum(a["input_tokens"] for a in priced),
            "reported_output_tokens": sum(a["output_tokens"] for a in priced),
            "estimated_provider_cost_usd": round(sum(a["estimated_usd"] for a in priced),10) if priced else None,
            "attempts_without_cost_data": len(attempts)-len(priced),
            "latency_ms": round((perf_counter()-start)*1000,3),
        },
    }
