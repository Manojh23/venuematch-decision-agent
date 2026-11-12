"""A bounded event-proposal graph. Inventory and prices come from the API."""
from datetime import date
from typing import Literal, TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, ConfigDict, Field, model_validator
from agent import Lead

Addon = Literal["av", "catering", "parking"]

class Item(BaseModel):
    item: str = Field(max_length=40)
    cents: int = Field(gt=0)

class InventorySlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,49}$")
    name: str = Field(max_length=100)
    city: str = Field(max_length=100)
    region: str = Field(pattern=r"^[A-Z]{2}$")
    capacity: int = Field(gt=0,le=1000)
    segments: list[str] = Field(max_length=4)
    amenities: list[Addon] = Field(max_length=3)
    day: date
    base_cents: int = Field(gt=0)
    total_cents: int = Field(gt=0)
    items: list[Item] = Field(min_length=1,max_length=4)

    @model_validator(mode="after")
    def verify_total(self):
        if self.total_cents!=sum(x.cents for x in self.items) or self.items[0].cents!=self.base_cents:
            raise ValueError("Inconsistent inventory pricing")
        return self

class EventRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    lead: Lead
    day: date
    guests: int = Field(ge=1,le=1000)
    addons: list[Addon] = Field(max_length=3)
    inventory: list[InventorySlot] = Field(max_length=60)

@tool
def filter_inventory(request: EventRequest) -> list[InventorySlot]:
    """Keep in-market inventory that fits segment, capacity, amenities, and date window."""
    return [slot for slot in request.inventory
            if slot.region==request.lead.region and request.lead.segment in slot.segments
            and slot.capacity>=request.guests and set(request.addons)<=set(slot.amenities)
            and abs((slot.day-request.day).days)<=3]

@tool
def rank_quotes(request: EventRequest, matches: list[InventorySlot]) -> list[dict]:
    """Compare verified quote totals and prefer dates and budgets that fit the event."""
    budget=request.lead.budget*100 if request.lead.budget is not None else None
    def score(slot):
        return (budget is not None and slot.total_cents>budget,
                abs((slot.day-request.day).days),slot.total_cents,slot.id,slot.day)
    selected=[]
    # First offer one date per venue, then expose more dates if inventory is narrow.
    ordered=sorted(matches,key=score)
    seen=set()
    for slot in ordered:
        if slot.id not in seen:
            selected.append(slot)
            seen.add(slot.id)
    selected.extend(slot for slot in ordered if slot not in selected)
    return [dict(slot.model_dump(mode="json"),
                 date_offset=(slot.day-request.day).days,
                 budget_status="unconfirmed" if budget is None else "within" if slot.total_cents<=budget else "over",
                 hold_allowed=budget is not None and slot.total_cents<=budget)
            for slot in selected[:5]]

class State(TypedDict,total=False):
    request: EventRequest
    qualification: dict
    matches: list[InventorySlot]
    options: list[dict]
    blockers: list[str]
    actions: list[str]

class ProposalPlanner:
    def __init__(self,qualifier):
        self.qualifier=qualifier
        builder=StateGraph(State)
        builder.add_node("qualify",lambda s:{"qualification":self.qualifier.run(s["request"].lead)})
        builder.add_node("filter_inventory",lambda s:{"matches":filter_inventory.invoke({"request":s["request"]})})
        builder.add_node("rank_quotes",lambda s:{"options":rank_quotes.invoke({"request":s["request"],"matches":s["matches"]})})
        builder.add_node("handoff",self.handoff)
        builder.add_edge(START,"qualify")
        builder.add_edge("qualify","filter_inventory")
        builder.add_edge("filter_inventory","rank_quotes")
        builder.add_edge("rank_quotes","handoff")
        builder.add_edge("handoff",END)
        self.graph=builder.compile()

    def handoff(self,state):
        request,options,q=state["request"],state["options"],state["qualification"]
        blockers,actions=[],[]
        if q["decision"]=="out_of_scope":
            blockers.append(q["reason"])
            actions.append("Review lead eligibility before sending a proposal.")
        if not options:
            blockers.append("No available venue meets the region, capacity, and amenity constraints.")
            actions.append("Try a different date, guest count, or add-on selection.")
        elif all(x["date_offset"]!=0 for x in options):
            blockers.append("No eligible venue is available on the requested date.")
            actions.append("Offer an alternate date and confirm it with the lead.")
        if request.lead.budget is None:
            blockers.append("Budget is unconfirmed.")
            actions.append("Ask the lead to confirm their budget before placing a hold.")
        elif options and all(x["budget_status"]=="over" for x in options):
            blockers.append("All matching quotes exceed the stated budget.")
            actions.append("Change add-ons or review a revised budget.")
        if q["decision"]=="review" and request.lead.budget is not None:
            blockers.append(q["reason"])
        if q["decision"]=="out_of_scope" or (q["decision"]=="review" and request.lead.budget is not None):
            for option in options:
                option["hold_allowed"]=False
        if any(o["hold_allowed"] for o in options):
            actions.append("Review the itemized quote, then place a temporary hold if appropriate.")
        return {"blockers":blockers,"actions":actions}

    def run(self,request):
        state=self.graph.invoke({"request":request})
        q=state["qualification"]
        return {**q,"options":state["options"],"blockers":state["blockers"],"actions":state["actions"],
                "path":q["path"]+["filter_inventory","rank_quotes","human_handoff"]}
