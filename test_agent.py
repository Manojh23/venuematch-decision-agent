import unittest
from uuid import uuid4
from fastapi.testclient import TestClient
from agent import Assessment, FixtureModel, Lead, Qualifier
from app import create_app
from evaluate import run

def lead(**changes):
    return Lead(**({"id":uuid4(),"segment":"corporate","region":"MA","budget":None,
                    "note":"Our group needs a conference venue in Boston."}|changes))

class AgentTests(unittest.TestCase):
    def test_fixture_regression(self):
        report=run()
        self.assertEqual((report["requests"],report["fixture_model_calls"],report["cache_hits"]),(30,3,6))

    def test_reject_invented_quote(self):
        class Invented(FixtureModel):
            def assess(self,note):
                return Assessment(direct_inquiry=True,quote="We already signed a contract"),{"source":"none","input":0,"output":0}
        answer=Qualifier(Invented()).run(lead())
        self.assertEqual((answer["decision"],answer["quote"]),("review",""))

    def test_cap_and_cache(self):
        agent=Qualifier(call_cap=1)
        record=lead()
        self.assertEqual(agent.run(record)["route"],"model")
        self.assertEqual(agent.run(record)["route"],"cache")
        self.assertEqual(agent.run(lead())["route"],"budget_fallback")

    def test_provider_error_does_not_cache(self):
        class Broken(FixtureModel):
            def assess(self,note): raise TimeoutError
        agent=Qualifier(Broken())
        record=lead()
        self.assertEqual(agent.run(record)["route"],"model_fallback")
        self.assertEqual(agent.run(record)["route"],"model_fallback")

    def test_http_validation_and_success(self):
        with TestClient(create_app(Qualifier())) as client:
            self.assertEqual(client.post("/qualify",json={}).status_code,422)
            res=client.post("/qualify",json=lead().model_dump(mode="json"))
            self.assertEqual(res.status_code,200)
            self.assertEqual(res.json()["quote"],"needs a conference venue")
