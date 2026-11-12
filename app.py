import os
from fastapi import FastAPI
from agent import Lead, Qualifier
from planner import EventRequest, ProposalPlanner
from routing import RoutedModel, load_specs
from sales import SalesSession, build_sales_queue


def create_app(qualifier=None):
    if qualifier is None:
        mode = os.getenv("AGENT_MODE","fixture")
        if mode not in {"fixture","routed"}:
            raise ValueError("AGENT_MODE must be fixture or routed.")
        model = RoutedModel(load_specs(os.environ["MODEL_ROUTES_JSON"])) if mode=="routed" else None
        cap = int(os.getenv("MODEL_CALL_CAP", "100"))
        if cap < 0:
            raise ValueError("MODEL_CALL_CAP must be nonnegative.")
        qualifier=Qualifier(model, call_cap=cap)
    planner=ProposalPlanner(qualifier)
    app=FastAPI(title="Sales Lead Qualification Agent",version="3.0")
    @app.get("/health")
    def health():
        return {"ok":True,"model":qualifier.model.label}
    @app.post("/qualify")
    def qualify(lead:Lead):
        return qualifier.run(lead)
    @app.post("/plan")
    def plan(request:EventRequest):
        return planner.run(request)
    @app.post("/sales/sessions")
    def sales_session(request:SalesSession):
        return build_sales_queue(request, qualifier)
    return app

app=create_app()
