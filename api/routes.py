from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from api.models import (
    SearchResponse,
    SuggestResponse,
    HealthResponse,
    Platform,
    SortOption
)
from agent.graph import agent
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY

# ──────────────────────────────────────────────────────
from cache import get_cached, set_cache

router = APIRouter()

llm = ChatGoogleGenerativeAI(
    api_key     = GEMINI_API_KEY,
    model       = "gemini-2.5-flash",
    max_retries = 0
)


# ─── Helper: Build Initial Agent State ───────────────

def build_initial_state(
    query:      str,
    platform:   str            = "all",
    sort:       str            = "price_asc",
    max_price:  Optional[int]  = None,
    min_price:  Optional[int]  = None,
    min_rating: Optional[float]= None,
    page:       int            = 1,
    limit:      int            = 100
) -> dict:
    return {
        "query":                    query,
        "parsed_intent":            {},
        "raw_results":              [],
        "final_results":            [],
        "filtered_results":         [],
        "filters": {
            "platform":   platform,
            "sort":       sort,
            "max_price":  max_price,
            "min_price":  min_price,
            "min_rating": min_rating,
        },
        "page":                     page,
        "limit":                    limit,
        "total_count":              0,
        "served_from_cache":        False,
        "execution_time":           None,
        "total_platforms_searched": 0,
        "platforms_failed":         [],
        "fallback_used":            False,
        "retry_count":              0,
        "suggestions":              [],
        "error":                    None
    }


# ─── Route 1: Health Check ────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check API health"
)
async def health_check():
    from cache import get_redis
    redis_status = "enabled" if get_redis() else "disabled"

    return HealthResponse(
        status  = "ok",
        cache   = redis_status,
        version = "1.0.0",
        message = f"Running with cache {redis_status}"
    )


# ─── Route 2: Search Products ─────────────────────────

@router.get(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search products across platforms"
)
async def search(
    q: str = Query(
        ...,
        description = "Search query",
        examples    = ["titan watch under 3000"]
    ),
    platform: Platform = Query(
        Platform.all,
        description = "Filter by platform"
    ),
    sort: SortOption = Query(
        SortOption.price_asc,
        description = "Sort results by"
    ),
    max_price: Optional[int] = Query(
        None,
        description = "Maximum price filter",
        examples    = [3000]
    ),
    min_price: Optional[int] = Query(
        None,
        description = "Minimum price filter",
        examples    = [500]
    ),
    min_rating: Optional[float] = Query(
        None,
        description = "Minimum rating filter",
        examples    = [4.0]
    ),
    page:  int = Query(1,  ge=1),
    limit: int = Query(100, ge=1, le=200)
):
    
    cache_key = (
        f"{q.strip().lower()}"
        f"|platform={platform.value}"
        f"|sort={sort.value}"
        f"|max={max_price}"
        f"|min={min_price}"
        f"|rating={min_rating}"
        f"|page={page}"
        f"|limit={limit}"
    )

    cached = get_cached(cache_key)
    if cached:
        return SearchResponse(**cached)

    # ── Run AI agent (only runs on cache miss) ────────
    try:
        state  = build_initial_state(
            query      = q,
            platform   = platform.value,
            sort       = sort.value,
            max_price  = max_price,
            min_price  = min_price,
            min_rating = min_rating,
            page       = page,
            limit      = limit
        )
        result = agent.invoke(state)

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Agent failed: {str(e)}"
        )

    if result.get("error"):
        raise HTTPException(
            status_code = 500,
            detail      = result["error"]
        )

    products = result.get("final_results", [])
    total    = result.get("total_count", len(products))

    response = SearchResponse(
        query   = q,
        total   = total,
        page    = page,
        results = products,
        source  = "live"
    )

    if products:
        set_cache(cache_key, response.dict())

    return response


# ─── Route 3: AI Powered Suggestions ─────────────────

@router.get(
    "/suggest",
    response_model=SuggestResponse,
    tags=["Search"],
    summary="Get AI powered search suggestions"
)
async def suggest(
    q: str = Query(
        ...,
        description = "Partial search query",
        examples    = ["lip stick"]
    ),
    limit: int = Query(5, ge=1, le=10)
):
    try:
        prompt = f"""
        Generate {limit} smart search suggestions for
        an Indian e-commerce price comparison website.

        User typed: "{q}"

        Rules:
        - Suggestions must be logically correct for the product
        - Consider who actually uses this product
        - Consider Indian market context
        - Consider common price ranges in India
        - Do NOT suggest irrelevant gender/age combinations
        - Keep suggestions concise and natural
        - Mix of: gender, age, occasion, price range, style

        Examples of GOOD suggestions:
        "lipstick" →
          ["lipstick under 500",
           "lipstick matte finish",
           "lipstick long lasting",
           "lipstick for daily use",
           "lipstick gift set"]

        "titan watch" →
          ["titan watch for men",
           "titan watch for women",
           "titan watch under 3000",
           "titan watch gifting",
           "titan watch formal"]

        "protein powder" →
          ["protein powder for gym",
           "protein powder under 2000",
           "protein powder whey",
           "protein powder for beginners",
           "protein powder chocolate flavour"]

        Now generate suggestions for: "{q}"

        Return ONLY a JSON array of strings.
        No explanation. No markdown. Just the array.
        Example: ["suggestion 1", "suggestion 2"]
        """

        response    = llm.invoke(prompt)
        raw_text    = response.content.strip()
        raw_text    = raw_text.replace(
                        "```json", ""
                      ).replace("```", "").strip()
        suggestions = json.loads(raw_text)

        if not isinstance(suggestions, list):
            raise ValueError("Not a list")

        suggestions = [
            str(s) for s in suggestions
            if isinstance(s, str) and s.strip()
        ][:limit]

    except Exception as e:
        print(f"[Suggest] AI failed: {str(e)}, using fallback")
        q_clean = q.strip()
        q_lower = q_clean.lower()

        phone_keywords  = ["phone", "mobile", "iphone", "samsung", "oneplus", "pixel", "realme", "redmi", "xiaomi", "motorola", "moto"]
        laptop_keywords = ["laptop", "macbook", "notebook", "asus", "dell", "hp", "lenovo", "acer"]

        if any(k in q_lower for k in phone_keywords):
            suggestions = [
                f"{q_clean} price",
                f"{q_clean} under 20000",
                f"{q_clean} under 50000",
                f"{q_clean} 5g",
                f"{q_clean} compare online"
            ][:limit]
        elif any(k in q_lower for k in laptop_keywords):
            suggestions = [
                f"{q_clean} price",
                f"{q_clean} under 40000",
                f"{q_clean} under 60000",
                f"{q_clean} for students",
                f"{q_clean} compare deals"
            ][:limit]
        else:
            suggestions = [
                f"{q_clean} price",
                f"{q_clean} compare",
                f"{q_clean} online",
                f"{q_clean} deals",
                f"{q_clean} store"
            ][:limit]

    return SuggestResponse(
        query       = q,
        suggestions = suggestions
    )


# ─── Route 4: Platforms List ──────────────────────────

@router.get(
    "/platforms",
    tags=["System"],
    summary="Get supported platforms"
)
async def get_platforms():
    return {
        "platforms": [
            {
                "id":    "all",
                "name":  "All Platforms",
                "color": "#0770e3"
            },
            {
                "id":    "amazon",
                "name":  "Amazon",
                "color": "#f59e0b"
            },
            {
                "id":    "flipkart",
                "name":  "Flipkart",
                "color": "#1d9e75"
            },
            {
                "id":    "myntra",
                "name":  "Myntra",
                "color": "#e24b4a"
            },
            {
                "id":    "ajio",
                "name":  "Ajio",
                "color": "#7f77dd"
            }
        ]
    }

# from fastapi import APIRouter, Query, HTTPException
# from typing import Optional
# from api.models import (
#     SearchResponse,
#     SuggestResponse,
#     HealthResponse,
#     Platform,
#     SortOption
# )
# from agent.graph import agent
# import json
# from langchain_google_genai import ChatGoogleGenerativeAI
# from config import GEMINI_API_KEY

# router = APIRouter()

# llm = ChatGoogleGenerativeAI(
#     api_key     = GEMINI_API_KEY,
#     model       = "gemini-2.5-flash",
#     max_retries = 0
# )


# # ─── Helper: Build Initial Agent State ───────────────

# def build_initial_state(
#     query:      str,
#     platform:   str            = "all",
#     sort:       str            = "price_asc",
#     max_price:  Optional[int]  = None,
#     min_price:  Optional[int]  = None,
#     min_rating: Optional[float]= None,
#     page:       int            = 1,
#     limit:      int            = 100
# ) -> dict:
#     return {
#         "query":                    query,
#         "parsed_intent":            {},
#         "raw_results":              [],
#         "final_results":            [],
#         "filtered_results":         [],
#         "filters": {
#             "platform":   platform,
#             "sort":       sort,
#             "max_price":  max_price,
#             "min_price":  min_price,
#             "min_rating": min_rating,
#         },
#         "page":                     page,
#         "limit":                    limit,
#         "total_count":              0,
#         "served_from_cache":        False,
#         "execution_time":           None,
#         "total_platforms_searched": 0,
#         "platforms_failed":         [],
#         "fallback_used":            False,
#         "retry_count":              0,
#         "suggestions":              [],
#         "error":                    None
#     }


# # ─── Route 1: Health Check ────────────────────────────

# @router.get(
#     "/health",
#     response_model=HealthResponse,
#     tags=["System"],
#     summary="Check API health"
# )
# async def health_check():
#     return HealthResponse(
#         status  = "ok",
#         cache   = "disabled",
#         version = "1.0.0",
#         message = "Running without cache"
#     )


# # ─── Route 2: Search Products ─────────────────────────

# @router.get(
#     "/search",
#     response_model=SearchResponse,
#     tags=["Search"],
#     summary="Search products across platforms"
# )
# async def search(
#     q: str = Query(
#         ...,
#         description = "Search query",
#         examples    = ["titan watch under 3000"]
#     ),
#     platform: Platform = Query(
#         Platform.all,
#         description = "Filter by platform"
#     ),
#     sort: SortOption = Query(
#         SortOption.price_asc,
#         description = "Sort results by"
#     ),
#     max_price: Optional[int] = Query(
#         None,
#         description = "Maximum price filter",
#         examples    = [3000]
#     ),
#     min_price: Optional[int] = Query(
#         None,
#         description = "Minimum price filter",
#         examples    = [500]
#     ),
#     min_rating: Optional[float] = Query(
#         None,
#         description = "Minimum rating filter",
#         examples    = [4.0]
#     ),
#     page:  int = Query(1,  ge=1),
#     limit: int = Query(100, ge=1, le=200)
# ):
#     # ── Run AI agent — no cache ──
#     try:
#         state  = build_initial_state(
#             query      = q,
#             platform   = platform.value,
#             sort       = sort.value,
#             max_price  = max_price,
#             min_price  = min_price,
#             min_rating = min_rating,
#             page       = page,
#             limit      = limit
#         )
#         result = agent.invoke(state)

#     except Exception as e:
#         raise HTTPException(
#             status_code = 500,
#             detail      = f"Agent failed: {str(e)}"
#         )

#     if result.get("error"):
#         raise HTTPException(
#             status_code = 500,
#             detail      = result["error"]
#         )

#     products = result.get("final_results", [])
#     total    = result.get("total_count", len(products))

#     return SearchResponse(
#         query   = q,
#         total   = total,
#         page    = page,
#         results = products,
#         source  = "live"
#     )


# # ─── Route 3: AI Powered Suggestions ─────────────────

# @router.get(
#     "/suggest",
#     response_model=SuggestResponse,
#     tags=["Search"],
#     summary="Get AI powered search suggestions"
# )
# async def suggest(
#     q: str = Query(
#         ...,
#         description = "Partial search query",
#         examples    = ["lip stick"]
#     ),
#     limit: int = Query(5, ge=1, le=10)
# ):
#     try:
#         prompt = f"""
#         Generate {limit} smart search suggestions for
#         an Indian e-commerce price comparison website.

#         User typed: "{q}"

#         Rules:
#         - Suggestions must be logically correct for the product
#         - Consider who actually uses this product
#         - Consider Indian market context
#         - Consider common price ranges in India
#         - Do NOT suggest irrelevant gender/age combinations
#         - Keep suggestions concise and natural
#         - Mix of: gender, age, occasion, price range, style

#         Examples of GOOD suggestions:
#         "lipstick" →
#           ["lipstick under 500",
#            "lipstick matte finish",
#            "lipstick long lasting",
#            "lipstick for daily use",
#            "lipstick gift set"]

#         "titan watch" →
#           ["titan watch for men",
#            "titan watch for women",
#            "titan watch under 3000",
#            "titan watch gifting",
#            "titan watch formal"]

#         "protein powder" →
#           ["protein powder for gym",
#            "protein powder under 2000",
#            "protein powder whey",
#            "protein powder for beginners",
#            "protein powder chocolate flavour"]

#         Now generate suggestions for: "{q}"

#         Return ONLY a JSON array of strings.
#         No explanation. No markdown. Just the array.
#         Example: ["suggestion 1", "suggestion 2"]
#         """

#         response    = llm.invoke(prompt)
#         raw_text    = response.content.strip()
#         raw_text    = raw_text.replace(
#                         "```json", ""
#                       ).replace("```", "").strip()
#         suggestions = json.loads(raw_text)

#         if not isinstance(suggestions, list):
#             raise ValueError("Not a list")

#         suggestions = [
#             str(s) for s in suggestions
#             if isinstance(s, str) and s.strip()
#         ][:limit]

#     except Exception as e:
#         print(f"[Suggest] AI failed: {str(e)}, using fallback")
#         q_clean = q.strip()
#         q_lower = q_clean.lower()
        
#         # Smart Logical Fallbacks based on category matching
#         phone_keywords = ["phone", "mobile", "iphone", "samsung", "oneplus", "pixel", "realme", "redmi", "xiaomi", "motorola", "moto"]
#         laptop_keywords = ["laptop", "macbook", "notebook", "asus", "dell", "hp", "lenovo", "acer"]
        
#         if any(k in q_lower for k in phone_keywords):
#             suggestions = [
#                 f"{q_clean} price",
#                 f"{q_clean} under 20000",
#                 f"{q_clean} under 50000",
#                 f"{q_clean} 5g",
#                 f"{q_clean} compare online"
#             ][:limit]
#         elif any(k in q_lower for k in laptop_keywords):
#             suggestions = [
#                 f"{q_clean} price",
#                 f"{q_clean} under 40000",
#                 f"{q_clean} under 60000",
#                 f"{q_clean} for students",
#                 f"{q_clean} compare deals"
#             ][:limit]
#         else:
#             suggestions = [
#                 f"{q_clean} price",
#                 f"{q_clean} compare",
#                 f"{q_clean} online",
#                 f"{q_clean} deals",
#                 f"{q_clean} store"
#             ][:limit]

#     return SuggestResponse(
#         query       = q,
#         suggestions = suggestions
#     )


# # ─── Route 4: Platforms List ──────────────────────────

# @router.get(
#     "/platforms",
#     tags=["System"],
#     summary="Get supported platforms"
# )
# async def get_platforms():
#     return {
#         "platforms": [
#             {
#                 "id":    "all",
#                 "name":  "All Platforms",
#                 "color": "#0770e3"
#             },
#             {
#                 "id":    "amazon",
#                 "name":  "Amazon",
#                 "color": "#f59e0b"
#             },
#             {
#                 "id":    "flipkart",
#                 "name":  "Flipkart",
#                 "color": "#1d9e75"
#             },
#             {
#                 "id":    "myntra",
#                 "name":  "Myntra",
#                 "color": "#e24b4a"
#             },
#             {
#                 "id":    "ajio",
#                 "name":  "Ajio",
#                 "color": "#7f77dd"
#             }
#         ]
#     }