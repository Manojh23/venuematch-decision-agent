"""Bounded lead qualification graph. All included records are synthetic."""
from collections import OrderedDict
from hashlib import sha256
from threading import Lock
from time import monotonic, perf_counter
from typing import Literal, TypedDict
from uuid import UUID

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

TARGETS = {"corporate": 5000, "wedding": 4000, "marine": 3000, "community": 3000}
REGIONS = {"MA", "RI"}
SIGNALS = ("need a venue", "need a waterfront venue", "need slip availability",
           "requesting a berth", "needs a conference venue", "looking for a wedding venue")
BLOCK = ("ignore all previous", "ignore previous", "system prompt", "reveal your instructions")


class Lead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    segment: Literal["corporate", "wedding", "marine", "community"]
    region: str = Field(pattern=r"^[A-Z]{2}$")
    budget: int | None = Field(ge=0, le=10000000)
    note: str = Field(min_length=1, max_length=600)


class Assessment(BaseModel):
    """Extract a direct booking request and verbatim evidence from a lead note."""
    direct_inquiry: bool
    quote: str = Field(max_length=240)


@tool
def structured_fit(segment: str, region: str, budget: int | None) -> str:
    """Apply business policy to market and stated budget, without a model."""
    if region not in REGIONS:
        return "out_of_market"
    if budget is None:
        return "missing_budget"
    return "meets_budget" if budget >= TARGETS[segment] else "below_budget"


@tool
def verify_quote(note: str, quote: str) -> bool:
    """Verify the quote is a verbatim nonempty span of the provided inquiry."""
    return len(quote.strip()) >= 12 and quote in note


class FixtureModel:
    label = "fixture"
    def assess(self, note):
        lower = note.lower()
        quote = next((note[i:i+len(s)] for s in SIGNALS if (i := lower.find(s)) >= 0), "")
        return Assessment(direct_inquiry=bool(quote), quote=quote), {
            "source":"synthetic_estimate", "input":max(1,len(note)//4), "output":max(1,len(quote)//4)
        }


class State(TypedDict, total=False):
    lead: Lead
    path: list[str]
    fit: str
    decision: str
    reason: str
    quote: str
    route: str
    usage: dict
    attempted: int
    provider_attempts: list[dict]
    assessment: Assessment


class Qualifier:
    def __init__(self, model=None, call_cap=100, ttl=300, input_rate=None, output_rate=None):
        self.model = model or FixtureModel()
        self.cap, self.ttl = call_cap, ttl
        self.calls = 0
        self.lock = Lock()
        self.cache = OrderedDict()
        self.input_rate, self.output_rate = input_rate, output_rate
        graph = StateGraph(State)
        for name, fn in (("policy",self.policy),("extract",self.extract),("validate",self.validate)):
            graph.add_node(name,fn)
        graph.add_edge(START,"policy")
        graph.add_conditional_edges("policy",lambda s:"extract" if s["route"]=="needs_model" else "done",{"extract":"extract","done":END})
        graph.add_conditional_edges("extract",lambda s:"validate" if "assessment" in s else "done",{"validate":"validate","done":END})
        graph.add_edge("validate",END)
        self.graph=graph.compile()

    def policy(self,state):
        lead=state["lead"]
        fit=structured_fit.invoke({"segment":lead.segment,"region":lead.region,"budget":lead.budget})
        base={"fit":fit,"path":["structured_fit"],"route":"rules"}
        if fit=="out_of_market":
            return {**base,"decision":"out_of_scope","reason":"Outside the configured markets."}
        if fit=="below_budget":
            return {**base,"decision":"out_of_scope","reason":"Stated budget is below the segment threshold."}
        if any(t in lead.note.lower() for t in BLOCK):
            return {**base,"decision":"review","reason":"Instruction-like content requires human review."}
        if fit=="meets_budget":
            return {**base,"decision":"shortlist","reason":"Market and stated budget meet the configured policy."}
        return {**base,"route":"needs_model"}

    def claim_call(self):
        with self.lock:
            if self.calls>=self.cap:
                return False
            self.calls+=1
            return True

    def extract(self,state):
        if hasattr(self.model, "assess_routed"):
            assessment, attempts, route = self.model.assess_routed(state["lead"].note, self.claim_call)
            trace = {"route":route,"provider_attempts":attempts,"attempted":len(attempts),
                     "path":state["path"]+["provider_routing"]}
            if assessment is None:
                return {**trace,"decision":"review","reason":
                        "Model-call quota reached." if route=="budget_fallback"
                        else "No provider returned a verified assessment."}
            return {**trace,"assessment":assessment}
        if not self.claim_call():
            return {"route":"budget_fallback","decision":"review","reason":"Model-call quota reached.","path":state["path"]+["quota_check"]}
        try:
            assessment,usage=self.model.assess(state["lead"].note)
        except Exception:
            return {"route":"model_fallback","decision":"review","reason":"Model returned no usable assessment.","path":state["path"]+["model_error"],"attempted":1}
        return {"route":"model","assessment":assessment,"usage":usage,"attempted":1,"path":state["path"]+["structured_extraction"]}

    def validate(self,state):
        answer=state["assessment"]
        matched=answer.direct_inquiry and verify_quote.invoke({"note":state["lead"].note,"quote":answer.quote})
        return {"decision":"review","quote":answer.quote if matched else "",
                "reason":"Direct inquiry verified; confirm budget with a human." if matched else "No verified direct inquiry with a stated budget.",
                "path":state["path"]+["verify_quote"]}

    def run(self,lead):
        start=perf_counter()
        fingerprint=getattr(self.model,"cache_key",self.model.label)
        key=sha256((fingerprint+"|v2|"+lead.model_dump_json()).encode()).hexdigest()
        with self.lock:
            cached=self.cache.get(key)
            if cached and cached[0]>monotonic():
                self.cache.move_to_end(key)
                return {**cached[1],"route":"cache","path":["cache_hit"],"attempted":0,"provider_attempts":[],"usage":{"source":"none","input":0,"output":0},"estimated_usd":None,"latency_ms":round((perf_counter()-start)*1000,3)}
        state=self.graph.invoke({"lead":lead})
        usage=state.get("usage",{"source":"none","input":0,"output":0})
        cost=None
        if usage["source"]=="provider" and self.input_rate is not None and self.output_rate is not None:
            cost=round((usage["input"]*self.input_rate+usage["output"]*self.output_rate)/1e6,8)
        attempts=state.get("provider_attempts",[])
        if attempts:
            measured=[x for x in attempts if x["estimated_usd"] is not None]
            usage={"source":"provider" if len(measured)==len(attempts) else "partial_provider" if measured else "unavailable",
                   "input":sum(x["input_tokens"] for x in measured),"output":sum(x["output_tokens"] for x in measured)}
            cost=round(sum(x["estimated_usd"] for x in measured),10) if len(measured)==len(attempts) else None
        result={k:state[k] for k in ("decision","reason","path","route")}
        result.update(quote=state.get("quote",""),attempted=state.get("attempted",0),provider_attempts=attempts,usage=usage,estimated_usd=cost,latency_ms=round((perf_counter()-start)*1000,3))
        if state["route"]=="model":
            with self.lock:
                self.cache[key]=(monotonic()+self.ttl,result)
                self.cache.move_to_end(key)
                while len(self.cache)>256:self.cache.popitem(last=False)
        return result
