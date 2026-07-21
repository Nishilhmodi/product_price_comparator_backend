from langgraph.graph import StateGraph, START, END
from .nodes import (
    intent_parser,
    search_node,
    aggregator,
    filter_node,
    response_formatter,
    error_handler
)
from .state import AgentState


# ─── Conditional Edge Functions ───────────────────────

def after_parser(state: AgentState) -> str:
    """Route after intent parser."""
    if state.get("error"):
        return "error"
    if not state.get("parsed_intent"):
        return "error"
    return "search"


def after_search(state: AgentState) -> str:
    """Route after search node."""
    if state.get("error"):
        return "error"

    results     = state.get("raw_results", [])
    retry_count = state.get("retry_count", 0)

    if len(results) == 0 and retry_count < 2:
        # Retry with simpler query — max 2 retries
        return "retry"

    # If results are empty but retries are exhausted, proceed to aggregator to return empty list gracefully
    return "aggregator"


def after_aggregator(state: AgentState) -> str:
    """Route after aggregator."""
    if state.get("error"):
        return "error"
    # Proceed to filter even if results are empty, to allow graceful formatting and filtering
    return "filter"


# ─── Build Graph ──────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # ── Register all nodes ──
    graph.add_node("parser",    intent_parser)
    graph.add_node("search",    search_node)
    graph.add_node("aggregator", aggregator)
    graph.add_node("filter",    filter_node)
    graph.add_node("formatter", response_formatter)
    graph.add_node("error",     error_handler)

    # ── Entry point ──
    graph.add_edge(START, "parser")

    # ── Conditional edges ──
    graph.add_conditional_edges(
        "parser",
        after_parser,
        {
            "search": "search",
            "error":  "error"
        }
    )

    graph.add_conditional_edges(
        "search",
        after_search,
        {
            "aggregator": "aggregator",
            "retry":      "search",      # loops back with higher retry_count
            "error":      "error"
        }
    )

    graph.add_conditional_edges(
        "aggregator",
        after_aggregator,
        {
            "filter": "filter",
            "error":  "error"
        }
    )

    # ── Straight edges ──
    graph.add_edge("filter",    "formatter")
    graph.add_edge("formatter", END)
    graph.add_edge("error",     END)

    return graph.compile()


# ── Single agent instance used across the app ──
agent = build_graph()