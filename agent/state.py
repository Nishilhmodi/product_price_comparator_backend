from typing import TypedDict, List, Optional


class AgentState(TypedDict):

    # ─── Core ─────────────────────────────────────────
    query:           str
    parsed_intent:   dict
    raw_results:     List[dict]
    final_results:   List[dict]
    error:           Optional[str]

    # ─── Filters ──────────────────────────────────────
    filters:          Optional[dict]
    filtered_results: List[dict]

    # ─── Pagination ───────────────────────────────────
    page:             int
    limit:            int
    total_count:      int

    # ─── Cache & Performance ──────────────────────────
    served_from_cache:        bool
    execution_time:           Optional[float]
    total_platforms_searched: int

    # ─── Search Metadata ──────────────────────────────
    platforms_failed: List[str]
    fallback_used:    bool
    retry_count:      int

    # ─── Suggestions ──────────────────────────────────
    suggestions:      List[str]