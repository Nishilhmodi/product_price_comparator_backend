from langgraph.graph import StateGraph, START, END

# ── FIX 1 ──────────────────────────────────────────────────────────────────
# BEFORE:
#   from .nodes import (
#       intent_parser,
#       search_node,       ← two separate nodes
#       aggregator,
#       filter_node,
#       response_formatter,
#       error_handler
#   )
#
# WHY IT WAS SLOW:
#   intent_parser (Gemini, ~8s) and search_node (SerpApi, ~12s) were two
#   separate graph nodes wired sequentially:
#       START → parser → search → aggregator → ...
#   This meant SerpApi could not start until Gemini fully finished.
#   Total wait = 8s + 12s = ~20s just for these two steps.
#
# WHAT WE CHANGED:
#   Merged both into a single node: parallel_intent_and_search.
#   Inside that node, Gemini and SerpApi run simultaneously using
#   ThreadPoolExecutor. The bottleneck is now only the slower of the two
#   (~12s SerpApi), not their sum.
#
#   Old flow:  parser(8s) → search(12s)  = ~20s sequential
#   New flow:  parallel_intent_and_search = ~12s parallel
#
# TIME SAVED: ~8-12 seconds per search.
# ───────────────────────────────────────────────────────────────────────────
from .nodes import (
    parallel_intent_and_search,   # ← replaces intent_parser + search_node
    aggregator,
    filter_node,
    response_formatter,
    error_handler
)
from .state import AgentState


# ─── Conditional Edge Functions ───────────────────────

# ── FIX 1 (continued) ──────────────────────────────────────────────────────
# BEFORE:
#   def after_parser(state) — routed from "parser" → "search" or "error"
#   def after_search(state) — routed from "search" → "aggregator"/"retry"/"error"
#
# WHAT WE CHANGED:
#   Removed after_parser entirely — it only existed to connect the two
#   sequential nodes. Now there is only after_parallel, which handles both
#   the error check AND the retry logic that after_search previously handled.
# ───────────────────────────────────────────────────────────────────────────

def after_parallel(state: AgentState) -> str:
    """
    Route after parallel_intent_and_search.
    Handles both the old after_parser check (error/no intent)
    and the old after_search check (retry on empty results).
    """
    if state.get("error"):
        return "error"

    # parsed_intent is always set by parallel node (fallback at minimum)
    if not state.get("parsed_intent"):
        return "error"

    results     = state.get("raw_results", [])
    retry_count = state.get("retry_count", 0)

    if len(results) == 0 and retry_count < 2:
        # Retry with a simpler query — max 2 retries
        # retry_count is incremented inside parallel_intent_and_search
        return "retry"

    # If results are empty but retries exhausted, proceed gracefully
    return "aggregator"


def after_aggregator(state: AgentState) -> str:
    """Route after aggregator — unchanged from original."""
    if state.get("error"):
        return "error"
    return "filter"


# ─── Build Graph ──────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # ── Register nodes ──
    # ── FIX 1: "parser" and "search" replaced by single "parallel_search" ──
    graph.add_node("parallel_search", parallel_intent_and_search)
    graph.add_node("aggregator",      aggregator)
    graph.add_node("filter",          filter_node)
    graph.add_node("formatter",       response_formatter)
    graph.add_node("error",           error_handler)

    # ── Entry point ──
    # BEFORE: graph.add_edge(START, "parser")
    # NOW:    goes directly into the combined parallel node
    graph.add_edge(START, "parallel_search")

    # ── Conditional edges ──
    graph.add_conditional_edges(
        "parallel_search",
        after_parallel,
        {
            "aggregator": "aggregator",
            "retry":      "parallel_search",  # loops back with higher retry_count
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

    # ── Straight edges — unchanged ──
    graph.add_edge("filter",    "formatter")
    graph.add_edge("formatter", END)
    graph.add_edge("error",     END)

    return graph.compile()


# ── Single agent instance used across the app ──
agent = build_graph()

# from langgraph.graph import StateGraph, START, END
# from .nodes import (
#     intent_parser,
#     search_node,
#     aggregator,
#     filter_node,
#     response_formatter,
#     error_handler
# )
# from .state import AgentState


# # ─── Conditional Edge Functions ───────────────────────

# def after_parser(state: AgentState) -> str:
#     """Route after intent parser."""
#     if state.get("error"):
#         return "error"
#     if not state.get("parsed_intent"):
#         return "error"
#     return "search"


# def after_search(state: AgentState) -> str:
#     """Route after search node."""
#     if state.get("error"):
#         return "error"

#     results     = state.get("raw_results", [])
#     retry_count = state.get("retry_count", 0)

#     if len(results) == 0 and retry_count < 2:
#         # Retry with simpler query — max 2 retries
#         return "retry"

#     # If results are empty but retries are exhausted, proceed to aggregator to return empty list gracefully
#     return "aggregator"


# def after_aggregator(state: AgentState) -> str:
#     """Route after aggregator."""
#     if state.get("error"):
#         return "error"
#     # Proceed to filter even if results are empty, to allow graceful formatting and filtering
#     return "filter"


# # ─── Build Graph ──────────────────────────────────────

# def build_graph():
#     graph = StateGraph(AgentState)

#     # ── Register all nodes ──
#     graph.add_node("parser",    intent_parser)
#     graph.add_node("search",    search_node)
#     graph.add_node("aggregator", aggregator)
#     graph.add_node("filter",    filter_node)
#     graph.add_node("formatter", response_formatter)
#     graph.add_node("error",     error_handler)

#     # ── Entry point ──
#     graph.add_edge(START, "parser")

#     # ── Conditional edges ──
#     graph.add_conditional_edges(
#         "parser",
#         after_parser,
#         {
#             "search": "search",
#             "error":  "error"
#         }
#     )

#     graph.add_conditional_edges(
#         "search",
#         after_search,
#         {
#             "aggregator": "aggregator",
#             "retry":      "search",      # loops back with higher retry_count
#             "error":      "error"
#         }
#     )

#     graph.add_conditional_edges(
#         "aggregator",
#         after_aggregator,
#         {
#             "filter": "filter",
#             "error":  "error"
#         }
#     )

#     # ── Straight edges ──
#     graph.add_edge("filter",    "formatter")
#     graph.add_edge("formatter", END)
#     graph.add_edge("error",     END)

#     return graph.compile()


# # ── Single agent instance used across the app ──
# agent = build_graph()