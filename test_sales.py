import unittest
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from agent import Assessment, FixtureModel, Qualifier
from app import create_app
from compare_routing import compare
from routing import ModelSpec, RoutedModel


def lead(budget, note="Our team needs a conference venue in Boston."):
    return {"id": str(uuid4()), "segment": "corporate", "region": "MA",
            "budget": budget, "note": note}


class Chat:
    def __init__(self, provider, calls):
        self.provider, self.calls = provider, calls

    def with_structured_output(self, schema, include_raw):
        return self

    def invoke(self, messages):
        self.calls.append(self.provider)
        if self.provider == "openai":
            raise TimeoutError
        quote = "We already signed a contract" if self.provider == "gemini" else "needs a conference venue"
        return {"raw": SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 20}),
                "parsed": Assessment(direct_inquiry=True, quote=quote)}


def configured(calls, cap=10):
    specs = [
        ModelSpec(provider="anthropic", model="fixture-a", input_per_million=3, output_per_million=12),
        ModelSpec(provider="gemini", model="fixture-g", input_per_million=.3, output_per_million=1.2),
        ModelSpec(provider="openai", model="fixture-o", input_per_million=.1, output_per_million=.4),
    ]
    return Qualifier(RoutedModel(specs, lambda s: Chat(s.provider, calls)), call_cap=cap)


class SalesTests(unittest.TestCase):
    def test_queue_routes_three_providers_and_reports_partial_cost(self):
        calls = []
        app = create_app(configured(calls))
        payload = {"leads": [lead(100), lead(None), lead(9000)]}
        with TestClient(app) as client:
            response = client.post("/sales/sessions", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            queue = response.json()
            self.assertEqual(calls, ["openai", "gemini", "anthropic"])
            self.assertEqual([row["decision"] for row in queue["leads"]],
                             ["shortlist", "review", "out_of_scope"])
            ambiguous = queue["leads"][1]
            self.assertEqual(ambiguous["quote"], "needs a conference venue")
            self.assertIn("confirm their budget", ambiguous["next_action"])
            self.assertEqual([a["status"] for a in ambiguous["provider_attempts"]],
                             ["provider_error", "unverified_evidence", "accepted"])
            self.assertEqual(queue["metrics"]["model_calls"], 3)
            self.assertEqual(queue["metrics"]["attempts_without_cost_data"], 1)
            self.assertEqual(queue["metrics"]["reported_input_tokens"], 200)
            self.assertAlmostEqual(queue["metrics"]["estimated_provider_cost_usd"], .000594)
            self.assertIsNone(ambiguous["estimated_usd"])
            repeated = client.post("/sales/sessions", json=payload).json()
            self.assertEqual(repeated["metrics"]["cache_hits"], 1)
            self.assertEqual(repeated["metrics"]["model_calls"], 0)
            self.assertIsNone(repeated["metrics"]["estimated_provider_cost_usd"])

    def test_fallbacks_share_quota_and_do_not_cache_failure(self):
        calls = []
        app = create_app(configured(calls, cap=2))
        with TestClient(app) as client:
            record = lead(None)
            first = client.post("/qualify", json=record).json()
            second = client.post("/qualify", json=record).json()
            self.assertEqual(calls, ["openai", "gemini"])
            self.assertEqual((first["route"], second["route"]), ("budget_fallback", "budget_fallback"))
            self.assertEqual((first["attempted"], second["attempted"]), (2, 0))

    def test_sales_session_rejects_duplicate_leads(self):
        with TestClient(create_app(Qualifier())) as client:
            row = lead(9000)
            self.assertEqual(client.post("/sales/sessions", json={"leads": [row, row]}).status_code, 422)

    def test_paired_comparison_requires_quality_and_usage(self):
        specs = [
            ModelSpec(provider="anthropic", model="baseline", input_per_million=4, output_per_million=8),
            ModelSpec(provider="gemini", model="routed", input_per_million=2, output_per_million=4),
        ]
        class Replay(Chat):
            def invoke(self, messages):
                answer, _ = FixtureModel().assess(messages[-1][1])
                return {"raw": SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 20}),
                        "parsed": answer}
        report = compare(specs, "baseline", lambda routes: RoutedModel(routes, lambda s: Replay(s.provider, [])))
        self.assertEqual(report["estimated_cost_ratio"], 2)
        self.assertEqual(report["multi_provider"]["passed"], 3)
        invalid = compare(specs, "baseline", lambda routes: RoutedModel(routes, lambda s: Chat(s.provider, [])))
        self.assertIsNone(invalid["estimated_cost_ratio"])
