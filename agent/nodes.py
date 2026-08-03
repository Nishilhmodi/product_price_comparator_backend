import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY
from scrapers.serpapi import search_products
from scrapers.platforms import parse_and_filter_results

llm = ChatGoogleGenerativeAI(
    api_key    = GEMINI_API_KEY,
    model      = "gemini-2.5-flash",
    max_retries= 0
)


# ─── Shared Intent Fallback ───────────────────────────

def _fallback_intent(raw_query: str) -> dict:
    """Return a safe default intent when Gemini fails."""
    return {
        "clean_query":    raw_query,
        "brand":          None,
        "category":       None,
        "budget":         None,
        "min_budget":     None,
        "gender":         None,
        "color":          None,
        "occasion":       None,
        "is_gift":        False,
        "sort_intent":    None,
        "query_language": "en"
    }


# ─── Node 1+2: Parallel Intent Parser + Search ────────
#
# Previously these were two sequential nodes:
#   intent_parser  (~8s Gemini)  →  search_node (~12s SerpApi)  = ~20s
#
# Now both run simultaneously in a ThreadPoolExecutor:
#   Gemini (8s)   ┐
#                 ├→ merge = ~12s  (bottleneck is whichever finishes last)
#   SerpApi (12s) ┘
#
# SerpApi search starts immediately using the raw query.
# When Gemini finishes, its clean_query is used to re-filter the
# already-fetched results — no extra API call needed.

def parallel_intent_and_search(state: dict) -> dict:
    """
    Runs Gemini intent parsing AND SerpApi search simultaneously.
    Saves 8-12 seconds vs running them sequentially.
    """
    start       = time.time()
    raw_query   = state["query"]
    retry_count = state.get("retry_count", 0)

    # ── Sub-task A: Gemini intent parsing ──────────────
    def run_intent() -> dict:
        try:
            prompt = f"""
        Extract search intent from this query: "{raw_query}"

        Return ONLY a JSON object with these exact keys:
        {{
            "clean_query":    "cleaned search term for Google Shopping",
            "brand":          "brand name or null",
            "category":       "product category or null",
            "budget":         max price as integer or null,
            "min_budget":     min price as integer or null,
            "gender":         "men" or "women" or "kids" or null,
            "color":          "color or null",
            "occasion":       "gifting/formal/casual or null",
            "is_gift":        true or false,
            "sort_intent":    "price_asc/price_desc/rating or null",
            "query_language": "en" or "hinglish"
        }}

        Examples:
        "titan watch under 3000"
        → clean_query: "Titan watch", budget: 3000

        "titan ghadi under 3k for dad"
        → clean_query: "Titan watch", budget: 3000,
          is_gift: true, query_language: "hinglish"

        "best samsung phone under 20000 with good camera"
        → clean_query: "Samsung smartphone",
          brand: "Samsung", budget: 20000

        Return JSON only. No explanation. No markdown.
        """

            response = llm.invoke(prompt)
            raw_text = response.content.strip()
            raw_text = raw_text.replace(
                "```json", ""
            ).replace("```", "").strip()

            return json.loads(raw_text)

        except json.JSONDecodeError:
            # LLM returned non-JSON — safe fallback
            return _fallback_intent(raw_query)

        except Exception as e:
            print(f"[Intent Parser Fallback] Gemini failed: {str(e)}")
            return _fallback_intent(raw_query)

    # ── Sub-task B: SerpApi search ─────────────────────
    # On retry, we don't have the Gemini intent yet, so we
    # use a simpler version of the raw query.
    def run_search() -> list:
        try:
            # First attempt: search with raw user query
            # Retry attempts: strip common noise words for simpler query
            search_query = raw_query
            if retry_count > 0:
                # Strip budget/price phrases for retry
                search_query = re.sub(
                    r'\b(under|above|below|upto|up to|less than|more than|around|'
                    r'Rs\.?|INR|₹)\s*[\d,k]+\b',
                    "", raw_query, flags=re.IGNORECASE
                ).strip()
                if not search_query:
                    search_query = raw_query

            return search_products(search_query, num=100)

        except Exception as e:
            print(f"[Search Sub-task Error]: {str(e)}")
            return []

    # ── Run both tasks in parallel ─────────────────────
    intent     = _fallback_intent(raw_query)  # safe default
    raw_results = []

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_intent = executor.submit(run_intent)
            future_search = executor.submit(run_search)

            # Collect results as they finish
            for future in as_completed(
                [future_intent, future_search], timeout=25
            ):
                if future is future_intent:
                    intent = future.result()
                else:
                    raw_results = future.result()

    except Exception as e:
        print(f"[Parallel Node Error]: {str(e)}")
        return {
            **state,
            "error":            f"Search failed: {str(e)}",
            "parsed_intent":    intent,
            "raw_results":      [],
            "platforms_failed": [],
            "execution_time":   round(time.time() - start, 2),
            "retry_count":      retry_count + 1
        }

    # ── Use Gemini's clean_query to re-filter results ──
    # SerpApi was called with the raw query, so now we
    # apply parse_and_filter_results using the cleaner
    # intent query for better relevance filtering.
    #
    # On retry, fall back to category if clean_query still
    # returns 0 results.
    filter_query = intent.get("clean_query") or raw_query
    if retry_count > 0:
        filter_query = (
            intent.get("category") or
            intent.get("clean_query") or
            raw_query
        )

    parsed = parse_and_filter_results(raw_results, query=filter_query)

    # If parse_and_filter_results returns 0 but there ARE raw results,
    # try again with the raw query as filter (looser match)
    if not parsed and raw_results:
        parsed = parse_and_filter_results(raw_results, query=raw_query)

    platforms_found  = set(r["platform"] for r in parsed)
    expected         = ["Amazon", "Flipkart", "Myntra", "Ajio"]
    platforms_failed = [
        p for p in expected if p not in platforms_found
    ]

    exec_time       = round(time.time() - start, 2)
    new_retry_count = (
        retry_count + 1 if len(parsed) == 0
        else retry_count
    )

    print(
        f"[Parallel Search] Done in {exec_time}s | "
        f"raw={len(raw_results)} parsed={len(parsed)} "
        f"retry={retry_count}"
    )

    return {
        **state,
        "parsed_intent":            intent,
        "raw_results":              parsed,
        "platforms_failed":         platforms_failed,
        "total_platforms_searched": len(platforms_found),
        "execution_time":           exec_time,
        "fallback_used":            retry_count > 0,
        "retry_count":              new_retry_count,
        "error":                    None
    }


# ─── Helper: Smart Product Key ────────────────────────

def get_product_key(title: str) -> str:
    """
    Extract a high-precision semantic key from product title.
    Uses an order-independent bag-of-words key model to prevent
    over-grouping of different products while accurately merging
    identical store listings.
    """
    title_lower = title.lower()

    # 1. Extract storage/memory (128gb, 256gb, 1tb)
    storage = ""
    storage_match = re.search(
        r'\b(\d+)\s*(gb|tb|mb)\b', title_lower
    )
    if storage_match:
        storage = storage_match.group(1) + storage_match.group(2)

    # 2. Extract volume/weight (30ml, 1kg, 9w)
    capacity = ""
    capacity_match = re.search(
        r'\b(\d+)\s*(ml|l|kg|g|w|watt)\b', title_lower
    )
    if capacity_match:
        capacity = (
            capacity_match.group(1) + capacity_match.group(2)
        )

    # 3. Tokenize
    tokens = re.findall(r'\b[a-z0-9-]+\b', title_lower)

    # Discard layout stopwords that carry no product model info
    layout_stopwords = {
        "with", "and", "the", "for", "from", "under", "in", "of", "to",
        "a", "an", "on", "at", "by", "or", "new", "original", "free",
        "delivery", "shipping", "warranty", "guarantee", "pack", "pcs"
    }

    filtered_parts = []
    for token in tokens:
        if token in layout_stopwords:
            continue
        if token == storage or token == capacity:
            continue
        filtered_parts.append(token)

    # Sort to make key order-independent (bag of words)
    filtered_parts.sort()

    key_parts = []
    seen = set()
    for p in filtered_parts:
        if p not in seen:
            seen.add(p)
            key_parts.append(p)

    if storage and storage not in seen:
        key_parts.append(storage)
    if capacity and capacity not in seen:
        key_parts.append(capacity)

    return "-".join(key_parts) if key_parts else title_lower


# ─── Node 3: Aggregator ───────────────────────────────

def aggregator(state: dict) -> dict:
    """
    Takes raw results list.
    Groups same product across platforms using smart key.
    Finds lowest price per product.
    Tags is_lowest flag.
    Sorts all products by lowest price.
    """
    try:
        products = state.get("raw_results", [])

        if not products:
            return {
                **state,
                "final_results": [],
                "total_count":   0
            }

        grouped = {}

        for item in products:
            if item["price"] <= 0:
                continue

            key = get_product_key(item["title"])

            if key not in grouped:
                grouped[key] = {
                    "title":    item["title"],
                    "image":    item["image"],
                    "brand":    state["parsed_intent"].get("brand", ""),
                    "category": state["parsed_intent"].get("category", ""),
                    "rating":   item["rating"],
                    "reviews":  item["reviews"],
                    "prices":   []
                }

            existing_platforms = [
                p["platform"] for p in grouped[key]["prices"]
            ]

            if item["platform"] not in existing_platforms:
                grouped[key]["prices"].append({
                    "platform":  item["platform"],
                    "price":     item["price"],
                    "url":       item["link"],
                    "is_lowest": False,
                    "discount":  item.get("discount")
                })

            if not grouped[key]["image"] and item["image"]:
                grouped[key]["image"] = item["image"]

            if grouped[key]["rating"] == 0 and item["rating"] > 0:
                grouped[key]["rating"] = item["rating"]

            if grouped[key]["reviews"] == 0 and item["reviews"] > 0:
                grouped[key]["reviews"] = item["reviews"]

        final = []
        for product in grouped.values():
            valid_prices = [
                p for p in product["prices"] if p["price"] > 0
            ]
            if not valid_prices:
                continue

            min_price = min(p["price"] for p in valid_prices)

            for p in product["prices"]:
                p["is_lowest"] = (p["price"] == min_price)

            product["prices"] = sorted(
                product["prices"],
                key=lambda x: x["price"] if x["price"] > 0 else 999999
            )

            product["lowest_price"] = min_price
            final.append(product)

        final = sorted(
            final,
            key=lambda x: x.get("lowest_price", 999999)
        )

    except Exception as e:
        return {
            **state,
            "error":         f"Aggregator failed: {str(e)}",
            "final_results": []
        }

    return {
        **state,
        "final_results": final,
        "total_count":   len(final),
        "error":         None
    }


# ─── Node 4: Filter Node ──────────────────────────────

def filter_node(state: dict) -> dict:
    try:
        results = state.get("final_results", [])
        filters = state.get("filters") or {}
        intent  = state.get("parsed_intent", {})

        platform = filters.get("platform", "all")
        if platform and platform != "all":
            results = [
                r for r in results
                if any(
                    p["platform"].lower() == platform.lower()
                    for p in r.get("prices", [])
                )
            ]

        max_price = filters.get("max_price") or intent.get("budget")
        min_price = filters.get("min_price") or intent.get("min_budget")

        if max_price:
            results = [
                r for r in results
                if r.get("lowest_price", 0) <= int(max_price)
            ]
        if min_price:
            results = [
                r for r in results
                if r.get("lowest_price", 0) >= int(min_price)
            ]

        min_rating = filters.get("min_rating")
        if min_rating:
            results = [
                r for r in results
                if r.get("rating", 0) >= float(min_rating)
            ]

        sort = (
            filters.get("sort") or
            intent.get("sort_intent") or
            "price_asc"
        )

        if sort == "price_asc":
            results = sorted(
                results,
                key=lambda x: x.get("lowest_price", 999999)
            )
        elif sort == "price_desc":
            results = sorted(
                results,
                key=lambda x: x.get("lowest_price", 0),
                reverse=True
            )
        elif sort == "rating":
            results = sorted(
                results,
                key=lambda x: x.get("rating", 0),
                reverse=True
            )

    except Exception as e:
        return {
            **state,
            "error":            f"Filter failed: {str(e)}",
            "filtered_results": state.get("final_results", [])
        }

    return {
        **state,
        "filtered_results": results,
        "total_count":      len(results),
        "error":            None
    }


# ─── Node 5: Response Formatter ──────────────────────

def response_formatter(state: dict) -> dict:
    try:
        results = (
            state.get("filtered_results") or
            state.get("final_results", [])
        )

        page  = state.get("page",  1)
        limit = state.get("limit", 100)
        start = (page - 1) * limit
        end   = start + limit

        paginated = results[start:end]

    except Exception as e:
        return {
            **state,
            "error":         f"Formatter failed: {str(e)}",
            "final_results": []
        }

    return {
        **state,
        "final_results": paginated,
        "total_count":   len(results),
        "error":         None
    }


# ─── Node 6: Error Handler ────────────────────────────

def error_handler(state: dict) -> dict:
    error_msg = state.get("error", "Unknown error")
    print(f"[PriceHunt Agent Error]: {error_msg}")

    return {
        **state,
        "final_results":    [],
        "filtered_results": [],
        "total_count":      0,
        "suggestions":      [],
        "error":            None
    }

# import json
# import time
# import re
# from langchain_google_genai import ChatGoogleGenerativeAI
# from config import GEMINI_API_KEY
# from scrapers.serpapi import search_products
# from scrapers.platforms import parse_and_filter_results

# llm = ChatGoogleGenerativeAI(
#     api_key    = GEMINI_API_KEY,
#     model      = "gemini-2.5-flash",
#     max_retries= 0
# )


# # ─── Node 1: Intent Parser ────────────────────────────

# def intent_parser(state: dict) -> dict:
#     try:
#         prompt = f"""
#         Extract search intent from this query: "{state['query']}"

#         Return ONLY a JSON object with these exact keys:
#         {{
#             "clean_query":    "cleaned search term for Google Shopping",
#             "brand":          "brand name or null",
#             "category":       "product category or null",
#             "budget":         max price as integer or null,
#             "min_budget":     min price as integer or null,
#             "gender":         "men" or "women" or "kids" or null,
#             "color":          "color or null",
#             "occasion":       "gifting/formal/casual or null",
#             "is_gift":        true or false,
#             "sort_intent":    "price_asc/price_desc/rating or null",
#             "query_language": "en" or "hinglish"
#         }}

#         Examples:
#         "titan watch under 3000"
#         → clean_query: "Titan watch", budget: 3000

#         "titan ghadi under 3k for dad"
#         → clean_query: "Titan watch", budget: 3000,
#           is_gift: true, query_language: "hinglish"

#         "best samsung phone under 20000 with good camera"
#         → clean_query: "Samsung smartphone",
#           brand: "Samsung", budget: 20000

#         Return JSON only. No explanation. No markdown.
#         """

#         response = llm.invoke(prompt)
#         raw_text = response.content.strip()
#         raw_text = raw_text.replace(
#             "```json", ""
#         ).replace("```", "").strip()

#         intent = json.loads(raw_text)

#     except json.JSONDecodeError:
#         intent = {
#             "clean_query":    state["query"],
#             "brand":          None,
#             "category":       None,
#             "budget":         None,
#             "min_budget":     None,
#             "gender":         None,
#             "color":          None,
#             "occasion":       None,
#             "is_gift":        False,
#             "sort_intent":    None,
#             "query_language": "en"
#         }

#     except Exception as e:
#         print(f"[Intent Parser Fallback] Failed: {str(e)}")
#         fallback_intent = {
#             "clean_query":    state["query"],
#             "brand":          None,
#             "category":       None,
#             "budget":         None,
#             "min_budget":     None,
#             "gender":         None,
#             "color":          None,
#             "occasion":       None,
#             "is_gift":        False,
#             "sort_intent":    None,
#             "query_language": "en"
#         }
#         return {
#             **state,
#             "parsed_intent": fallback_intent,
#             "error":         None
#         }

#     return {
#         **state,
#         "parsed_intent": intent,
#         "error":         None
#     }


# # ─── Node 2: Search Node ──────────────────────────────

# def search_node(state: dict) -> dict:
#     try:
#         start       = time.time()
#         intent      = state.get("parsed_intent", {})
#         query       = intent.get("clean_query") or state["query"]
#         retry_count = state.get("retry_count", 0)

#         if retry_count > 0:
#             query = intent.get("category") or state["query"]

#         # ── Single SerpApi call — 1 credit only ──
#         raw    = search_products(query, num=100)
#         parsed = parse_and_filter_results(raw, query=query)

#         platforms_found  = set(r["platform"] for r in parsed)
#         expected         = ["Amazon", "Flipkart", "Myntra", "Ajio"]
#         platforms_failed = [
#             p for p in expected if p not in platforms_found
#         ]

#         exec_time       = round(time.time() - start, 2)
#         new_retry_count = (
#             retry_count + 1 if len(parsed) == 0
#             else retry_count
#         )

#     except Exception as e:
#         print(f"[Search Node Error]: {str(e)}")
#         return {
#             **state,
#             "error":            f"Search failed: {str(e)}",
#             "raw_results":      [],
#             "platforms_failed": [],
#             "execution_time":   0.0,
#             "retry_count":      retry_count + 1
#         }

#     return {
#         **state,
#         "raw_results":              parsed,
#         "platforms_failed":         platforms_failed,
#         "total_platforms_searched": len(platforms_found),
#         "execution_time":           exec_time,
#         "fallback_used":            retry_count > 0,
#         "retry_count":              new_retry_count,
#         "error":                    None
#     }


# # ─── Helper: Smart Product Key ────────────────────────

# def get_product_key(title: str) -> str:
#     """
#     Extract a high-precision semantic key from product title.
#     Uses an order-independent bag-of-words key model to prevent over-grouping
#     of different products while accurately merging identical store listings.
#     """
#     title_lower = title.lower()

#     # 1. Extract storage/memory (128gb, 256gb, 1tb)
#     storage = ""
#     storage_match = re.search(
#         r'\b(\d+)\s*(gb|tb|mb)\b', title_lower
#     )
#     if storage_match:
#         storage = storage_match.group(1) + storage_match.group(2)

#     # 2. Extract volume/weight (30ml, 1kg, 9w)
#     capacity = ""
#     capacity_match = re.search(
#         r'\b(\d+)\s*(ml|l|kg|g|w|watt)\b', title_lower
#     )
#     if capacity_match:
#         capacity = (
#             capacity_match.group(1) + capacity_match.group(2)
#         )

#     # 3. Tokenize
#     tokens = re.findall(r'\b[a-z0-9-]+\b', title_lower)

#     # Discard simple layout stopwords that add no product model information
#     layout_stopwords = {
#         "with", "and", "the", "for", "from", "under", "in", "of", "to", 
#         "a", "an", "on", "at", "by", "or", "new", "original", "free", 
#         "delivery", "shipping", "warranty", "guarantee", "pack", "pcs"
#     }

#     filtered_parts = []
#     for token in tokens:
#         if token in layout_stopwords:
#             continue
#         if token == storage or token == capacity:
#             continue
#         filtered_parts.append(token)

#     # Sort parts to make the key order-independent (bag of words)
#     filtered_parts.sort()

#     key_parts = []
#     # Deduplicate preserving sorted order
#     seen = set()
#     for p in filtered_parts:
#         if p not in seen:
#             seen.add(p)
#             key_parts.append(p)

#     if storage and storage not in seen:
#         key_parts.append(storage)
#     if capacity and capacity not in seen:
#         key_parts.append(capacity)

#     return "-".join(key_parts) if key_parts else title_lower


# # ─── Node 3: Aggregator ───────────────────────────────

# def aggregator(state: dict) -> dict:
#     """
#     Takes raw results list.
#     Groups same product across platforms using smart key.
#     Finds lowest price per product.
#     Tags is_lowest flag.
#     Sorts all products by lowest price.
#     """
#     try:
#         products = state.get("raw_results", [])

#         if not products:
#             return {
#                 **state,
#                 "final_results": [],
#                 "total_count":   0
#             }

#         grouped = {}

#         for item in products:
#             if item["price"] <= 0:
#                 continue

#             key = get_product_key(item["title"])

#             if key not in grouped:
#                 grouped[key] = {
#                     "title":    item["title"],
#                     "image":    item["image"],
#                     "brand":    state["parsed_intent"].get("brand", ""),
#                     "category": state["parsed_intent"].get("category", ""),
#                     "rating":   item["rating"],
#                     "reviews":  item["reviews"],
#                     "prices":   []
#                 }

#             existing_platforms = [
#                 p["platform"] for p in grouped[key]["prices"]
#             ]

#             if item["platform"] not in existing_platforms:
#                 grouped[key]["prices"].append({
#                     "platform":  item["platform"],
#                     "price":     item["price"],
#                     "url":       item["link"],
#                     "is_lowest": False,
#                     "discount":  item.get("discount")
#                 })

#             if not grouped[key]["image"] and item["image"]:
#                 grouped[key]["image"] = item["image"]

#             if grouped[key]["rating"] == 0 and item["rating"] > 0:
#                 grouped[key]["rating"] = item["rating"]

#             if grouped[key]["reviews"] == 0 and item["reviews"] > 0:
#                 grouped[key]["reviews"] = item["reviews"]

#         final = []
#         for product in grouped.values():
#             valid_prices = [
#                 p for p in product["prices"] if p["price"] > 0
#             ]
#             if not valid_prices:
#                 continue

#             min_price = min(p["price"] for p in valid_prices)

#             for p in product["prices"]:
#                 p["is_lowest"] = (p["price"] == min_price)

#             product["prices"] = sorted(
#                 product["prices"],
#                 key=lambda x: x["price"] if x["price"] > 0 else 999999
#             )

#             product["lowest_price"] = min_price
#             final.append(product)

#         final = sorted(
#             final,
#             key=lambda x: x.get("lowest_price", 999999)
#         )

#         # ← resolve_single_product_offers REMOVED
#         # Previously this called SerpApi N times (1 per product)
#         # causing N+1 credits per search
#         # Links are already handled by extract_direct_link

#     except Exception as e:
#         return {
#             **state,
#             "error":         f"Aggregator failed: {str(e)}",
#             "final_results": []
#         }

#     return {
#         **state,
#         "final_results": final,
#         "total_count":   len(final),
#         "error":         None
#     }


# # ─── Node 4: Filter Node ──────────────────────────────

# def filter_node(state: dict) -> dict:
#     try:
#         results = state.get("final_results", [])
#         filters = state.get("filters") or {}
#         intent  = state.get("parsed_intent", {})

#         platform = filters.get("platform", "all")
#         if platform and platform != "all":
#             results = [
#                 r for r in results
#                 if any(
#                     p["platform"].lower() == platform.lower()
#                     for p in r.get("prices", [])
#                 )
#             ]

#         max_price = filters.get("max_price") or intent.get("budget")
#         min_price = filters.get("min_price") or intent.get("min_budget")

#         if max_price:
#             results = [
#                 r for r in results
#                 if r.get("lowest_price", 0) <= int(max_price)
#             ]
#         if min_price:
#             results = [
#                 r for r in results
#                 if r.get("lowest_price", 0) >= int(min_price)
#             ]

#         min_rating = filters.get("min_rating")
#         if min_rating:
#             results = [
#                 r for r in results
#                 if r.get("rating", 0) >= float(min_rating)
#             ]

#         sort = (
#             filters.get("sort") or
#             intent.get("sort_intent") or
#             "price_asc"
#         )

#         if sort == "price_asc":
#             results = sorted(
#                 results,
#                 key=lambda x: x.get("lowest_price", 999999)
#             )
#         elif sort == "price_desc":
#             results = sorted(
#                 results,
#                 key=lambda x: x.get("lowest_price", 0),
#                 reverse=True
#             )
#         elif sort == "rating":
#             results = sorted(
#                 results,
#                 key=lambda x: x.get("rating", 0),
#                 reverse=True
#             )

#     except Exception as e:
#         return {
#             **state,
#             "error":            f"Filter failed: {str(e)}",
#             "filtered_results": state.get("final_results", [])
#         }

#     return {
#         **state,
#         "filtered_results": results,
#         "total_count":      len(results),
#         "error":            None
#     }


# # ─── Node 5: Response Formatter ──────────────────────

# def response_formatter(state: dict) -> dict:
#     try:
#         results = (
#             state.get("filtered_results") or
#             state.get("final_results", [])
#         )

#         page  = state.get("page",  1)
#         limit = state.get("limit", 100)
#         start = (page - 1) * limit
#         end   = start + limit

#         paginated = results[start:end]

#     except Exception as e:
#         return {
#             **state,
#             "error":         f"Formatter failed: {str(e)}",
#             "final_results": []
#         }

#     return {
#         **state,
#         "final_results": paginated,
#         "total_count":   len(results),
#         "error":         None
#     }


# # ─── Node 6: Error Handler ────────────────────────────

# def error_handler(state: dict) -> dict:
#     error_msg = state.get("error", "Unknown error")
#     print(f"[PriceHunt Agent Error]: {error_msg}")

#     return {
#         **state,
#         "final_results":    [],
#         "filtered_results": [],
#         "total_count":      0,
#         "suggestions":      [],
#         "error":            None
#     }