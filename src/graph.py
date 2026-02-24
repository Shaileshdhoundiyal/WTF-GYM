import os
from langchain_core.messages import BaseMessage, AIMessage
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from src.agents.main_agent import main_agent
from src.agents.exercise_specialist_agent import exercise_specialist_agent
from src.agents.diet_specialist_agent import diet_specialist_agent
from src.agents.myth_buster_agent import myth_buster_agent

load_dotenv()


# ── State Definition ───────────────────────────────────────────────────────
class State(TypedDict):
    messages: List[BaseMessage]
    current_agent: str
    next: str                    # Routing signal for conditional edges

    # Exercise specialist fields
    fitness_goal: str
    target_muscles: str
    experience_level: str

    # Diet specialist fields
    dietary_goal: str
    dietary_restrictions: str

    # Myth buster fields
    myth_claim: str

    # Shared specialist output
    specialist_response: str


# ── Node Wrappers ──────────────────────────────────────────────────────────
def run_main_agent(state: State) -> Dict:
    return main_agent(state)


# ── Router ─────────────────────────────────────────────────────────────────
def router(state: State) -> str:
    next_agent = state.get("next", "")
    if next_agent == "exercise_specialist":
        return "exercise_specialist"
    elif next_agent == "diet_specialist":
        return "diet_specialist"
    elif next_agent == "myth_buster":
        return "myth_buster"
    return END


# ── Build the Graph ────────────────────────────────────────────────────────
workflow = StateGraph(State)

# Add nodes
workflow.add_node("main", run_main_agent)
workflow.add_node("exercise_specialist", exercise_specialist_agent)
workflow.add_node("diet_specialist", diet_specialist_agent)
workflow.add_node("myth_buster", myth_buster_agent)

# Entry point
workflow.set_entry_point("main")

# Conditional edges from main → specialists or END
workflow.add_conditional_edges(
    "main",
    router,
    {
        "exercise_specialist": "exercise_specialist",
        "diet_specialist": "diet_specialist",
        "myth_buster": "myth_buster",
        END: END,
    }
)

# All specialists → END (their response is the final output)
workflow.add_edge("exercise_specialist", END)
workflow.add_edge("diet_specialist", END)
workflow.add_edge("myth_buster", END)

# Compile
app_graph = workflow.compile()
