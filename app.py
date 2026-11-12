import os
from math import isfinite
from fastapi import FastAPI
from agent import ClaudeModel, Lead, Qualifier
from planner import EventRequest, ProposalPlanner


def rate(name):
    if name not in os.environ:
        return None
    value = float(os.environ[name])
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def create_app(qualifier=None):
    if qualifier is None:
        mode = os.getenv("AGENT_MODE","fixture")
        if mode not in {"fixture","claude"}:
            raise ValueError("AGENT_MODE must be fixture or claude.")
        qualifier=Qualifier(ClaudeModel(os.environ["CLAUDE_MODEL"]) if mode=="claude" else None,
                           call_cap=int(os.getenv("MODEL_CALL_CAP","100")),
                           input_rate=rate("INPUT_USD_PER_MTOK"),
                           output_rate=rate("OUTPUT_USD_PER_MTOK"))
    planner=ProposalPlanner(qualifier)
    app=FastAPI(title="VenueMatch Decision Agent",version="2.0")
    @app.get("/health")
    def health():
        return {"ok":True,"model":qualifier.model.label}
    @app.post("/qualify")
    def qualify(lead:Lead):
        return qualifier.run(lead)
    @app.post("/plan")
    def plan(request:EventRequest):
        return planner.run(request)
    return app

app=create_app()
