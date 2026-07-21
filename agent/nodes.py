import json
import time
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY
from scrapers.serpapi import search_products
from scrapers.platforms import parse_and_filter_results

llm = ChatGoogleGenerativeAI(
    api_key    = GEMINI_API_KEY,
    model      = "gemini-2.5-flash",
    max_retries= 0
)


# ─── Node 1: Intent Parser ────────────────────────────

def intent_parser(state: dict) -> dict:
    try:
        prompt = f"""
        Extract search intent from this query: "{state['query']}"

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

        intent = json.loads(raw_text)

    except json.JSONDecodeError:
        intent = {
            "clean_query":    state["query"],
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

    except Exception as e:
        print(f"[Intent Parser Fallback] Failed: {str(e)}")
        fallback_intent = {
            "clean_query":    state["query"],
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
        return {
            **state,
            "parsed_intent": fallback_intent,
            "error":         None
        }

    return {
        **state,
        "parsed_intent": intent,
        "error":         None
    }


# ─── Node 2: Search Node ──────────────────────────────

def search_node(state: dict) -> dict:
    try:
        start       = time.time()
        intent      = state.get("parsed_intent", {})
        query       = intent.get("clean_query") or state["query"]
        retry_count = state.get("retry_count", 0)

        if retry_count > 0:
            query = intent.get("category") or state["query"]

        # ── Single SerpApi call — 1 credit only ──
        raw    = search_products(query, num=100)
        parsed = parse_and_filter_results(raw, query=query)

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

    except Exception as e:
        print(f"[Search Node Error]: {str(e)}")
        return {
            **state,
            "error":            f"Search failed: {str(e)}",
            "raw_results":      [],
            "platforms_failed": [],
            "execution_time":   0.0,
            "retry_count":      retry_count + 1
        }

    return {
        **state,
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
    Uses an order-independent bag-of-words key model to prevent over-grouping
    of different products while accurately merging identical store listings.
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

    # Discard simple layout stopwords that add no product model information
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

    # Sort parts to make the key order-independent (bag of words)
    filtered_parts.sort()

    key_parts = []
    # Deduplicate preserving sorted order
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

        # ← resolve_single_product_offers REMOVED
        # Previously this called SerpApi N times (1 per product)
        # causing N+1 credits per search
        # Links are already handled by extract_direct_link

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
# from scrapers.platforms import parse_result, parse_and_filter_results
# from concurrent.futures import ThreadPoolExecutor
# import requests
# import hashlib
# from cache import get_cached, set_cache
# from config import SERPAPI_KEY
# from scrapers.platforms import normalize_platform, parse_price, extract_destination_url

# llm = ChatGoogleGenerativeAI(
#     api_key=GEMINI_API_KEY,
#     model="gemini-2.5-flash",
#     max_retries=0
# )


# # ─── Node 1: Intent Parser ────────────────────────────

# def intent_parser(state: dict) -> dict:
#     """
#     Reads raw user query.
#     Uses Groq LLM to extract brand, budget, gender etc.
#     Returns clean structured intent.
#     """
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

#         # Clean markdown fences if model adds them
#         raw_text = raw_text.replace(
#             "```json", ""
#         ).replace("```", "").strip()

#         intent = json.loads(raw_text)

#     except json.JSONDecodeError:
#         # LLM returned non-JSON — use raw query as fallback
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
#         print(f"[Intent Parser Fallback] Groq failed: {str(e)}")
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

# # ── Update search_node ──
# def search_node(state: dict) -> dict:
#     try:
#         start       = time.time()
#         intent      = state.get("parsed_intent", {})
#         query       = intent.get("clean_query") or state["query"]
#         retry_count = state.get("retry_count", 0)

#         if retry_count > 0:
#             query = intent.get("category") or state["query"]

#         raw    = search_products(query, num=100)

#         # ← changed from [parse_result(r) for r in raw]
#         parsed = parse_and_filter_results(raw)

#         platforms_found  = set(r["platform"] for r in parsed)
#         expected         = ["Amazon", "Flipkart", "Myntra", "Ajio"]
#         platforms_failed = [
#             p for p in expected if p not in platforms_found
#         ]

#         exec_time = round(time.time() - start, 2)
#         new_retry_count = retry_count + 1 if len(parsed) == 0 else retry_count

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
#     Extract a high-precision semantic key from the product title.
#     Ensures that different models, sizes, capacities, and specs are NOT grouped together.
#     """
#     title_lower = title.lower()
    
#     # 1. Extract storage/memory capacity (e.g. 128gb, 256gb, 512gb, 1tb)
#     storage = ""
#     storage_match = re.search(r'\b(\d+)\s*(gb|tb|mb)\b', title_lower)
#     if storage_match:
#         storage = storage_match.group(1) + storage_match.group(2)
        
#     # 2. Extract volume/weight (e.g. 30ml, 50ml, 100ml, 1kg, 2kg, 9w, 12w)
#     capacity = ""
#     capacity_match = re.search(r'\b(\d+)\s*(ml|l|kg|g|w|watt)\b', title_lower)
#     if capacity_match:
#         capacity = capacity_match.group(1) + capacity_match.group(2)

#     # 3. Extract tokens and filter them
#     tokens = re.findall(r'\b[a-z0-9-]+\b', title_lower)
    
#     # Define a list of generic words to discard
#     discard_words = {
#         "with", "and", "the", "for", "from", "under", "new", "original",
#         "dial", "strap", "band", "wrist", "wristwatch", "waterproof", "resistant",
#         "stylish", "elegant", "premium", "quality", "fashion", "collection",
#         "analog", "digital", "quartz", "quartz-powered", "movement", "display",
#         "round", "square", "rectangle", "oval", "tonneau", "shape",
#         "silver", "gold", "black", "blue", "green", "brown", "white", "red", "grey", "pink",
#         "leather", "steel", "stainless", "silicone", "rubber", "mesh", "chain",
#         "clasp", "buckle", "bezel", "glass", "crystal", "mineral", "sapphire",
#         "free", "delivery", "shipping", "warranty", "guarantee", "pack", "of", "pcs"
#     }
    
#     key_parts = []
    
#     # Extract brand first for ordering
#     brands = ["apple", "samsung", "nike", "puma", "timex", "sony", "philips", "loreal", "l'oreal", "titan", "casio", "fastrack"]
#     detected_brand = ""
#     for b in brands:
#         if b in title_lower:
#             detected_brand = b
#             break
            
#     if detected_brand:
#         key_parts.append(detected_brand)
        
#     # Specific model keywords we want to preserve at all costs
#     preserve_keywords = {
#         # Phones/Tech
#         "pro", "max", "plus", "ultra", "lite", "mini", "air", "active", "watch", "phone", "galaxy", "iphone",
#         # Watches
#         "chronograph", "automatic", "mechanical", "multifunction", "weekender", "classics", "karishma", "expedition", "waterbury", "ironman",
#         # Sneakers/Shoes
#         "sneakers", "running", "sports", "dry-fit", "t-shirt", "shirt", "shoes", "shoe",
#         # Headphones
#         "headphones", "earbuds", "wireless", "noise", "cancelling",
#         # Home/Bulbs
#         "hue", "smart", "led", "bulb",
#         # Beauty
#         "hyaluronic", "acid", "face", "serum", "shampoo", "conditioner"
#     }
    
#     for token in tokens:
#         # Skip brand since we already added it
#         if token == detected_brand or (detected_brand == "loreal" and token == "l'oreal"):
#             continue
            
#         # If it's a discard word, skip it
#         if token in discard_words:
#             continue
            
#         # If it contains numbers (like s24, 15, xm5, 1000xm5), we definitely want it!
#         if any(char.isdigit() for char in token):
#             # Check if it matches the storage or capacity we already extracted, to avoid duplication
#             if token == storage or token == capacity:
#                 continue
#             key_parts.append(token)
#             continue
            
#         # If it's a preserve keyword, we want it!
#         if token in preserve_keywords:
#             key_parts.append(token)
#             continue
            
#         # If it's not a generic word, and it's long enough, it might be a model name
#         if len(token) > 3 and token.isalpha():
#             key_parts.append(token)
            
#     # Add storage and capacity to the end of the key for distinctness
#     if storage:
#         key_parts.append(storage)
#     if capacity:
#         key_parts.append(capacity)
        
#     # Deduplicate while preserving order
#     seen = set()
#     unique_parts = []
#     for p in key_parts:
#         if p not in seen:
#             seen.add(p)
#             unique_parts.append(p)
            
#     return " ".join(unique_parts)


# def resolve_single_product_offers(product: dict) -> dict:
#     """
#     Resolves direct merchant offers for a single product from SerpApi or Redis cache.
#     Modifies the product dict in-place and returns it.
#     """
#     token = product.get("immersive_token")
#     product_id = product.get("product_id")
    
#     if not token and not product_id:
#         product["resolved"] = True
#         return product

#     # 1. Check cache
#     key_src = token if token else product_id
#     key_hash = hashlib.md5(key_src.encode('utf-8')).hexdigest()
#     cache_key = f"offers:{key_hash}"

#     try:
#         cached = get_cached(cache_key)
#         if cached and "offers" in cached:
#             # We found cached offers! Update prices
#             product["prices"] = cached["offers"]
#             # Find the lowest price offer
#             if cached["offers"]:
#                 min_price = min(o["price"] for o in cached["offers"])
#                 for o in product["prices"]:
#                     o["is_lowest"] = (o["price"] == min_price)
#                 product["lowest_price"] = min_price
#             product["resolved"] = True
#             return product
#     except Exception:
#         pass

#     # 2. Query SerpApi google_immersive_product
#     try:
#         params = {
#             "engine": "google_immersive_product",
#             "gl": "in",
#             "hl": "en",
#             "api_key": SERPAPI_KEY
#         }
#         if token:
#             params["page_token"] = token
#         else:
#             params["product_id"] = product_id

#         # Use a short timeout of 5 seconds to prevent hanging
#         response = requests.get(
#             "https://serpapi.com/search",
#             params=params,
#             timeout=5
#         )
#         response.raise_for_status()
#         data = response.json()

#         product_results = data.get("product_results", {})
#         stores = product_results.get("stores", [])

#         allowed_platforms = {"Amazon", "Flipkart", "Myntra", "Ajio"}
#         offers_dict = {}

#         for s in stores:
#             name = s.get("name", "")
#             norm_name = normalize_platform(name)
#             if norm_name not in allowed_platforms:
#                 continue

#             price_val = parse_price(s.get("price", "0"))
#             if price_val <= 0:
#                 continue

#             raw_url = s.get("link", "")
#             direct_url = extract_destination_url(raw_url)

#             if norm_name in offers_dict:
#                 if price_val < offers_dict[norm_name]["price"]:
#                     offers_dict[norm_name] = {
#                         "platform": norm_name,
#                         "price": price_val,
#                         "url": direct_url,
#                         "is_lowest": False
#                     }
#             else:
#                 offers_dict[norm_name] = {
#                     "platform": norm_name,
#                     "price": price_val,
#                     "url": direct_url,
#                     "is_lowest": False
#                 }

#         offers_list = list(offers_dict.values())
#         if offers_list:
#             min_price = min(o["price"] for o in offers_list)
#             for o in offers_list:
#                 o["is_lowest"] = (o["price"] == min_price)

#             product["prices"] = offers_list
#             product["lowest_price"] = min_price

#             # Save to cache
#             try:
#                 set_cache(cache_key, {
#                     "offers": offers_list
#                 })
#             except Exception:
#                 pass

#         product["resolved"] = True

#     except Exception as e:
#         print(f"[Backend Resolver] Failed for product '{product.get('title')}': {str(e)}")
#         product["resolved"] = False

#     return product


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
#             # Skip items with no price
#             if item["price"] <= 0:
#                 continue

#             # Smart grouping key
#             key = get_product_key(item["title"])

#             if key not in grouped:
#                 grouped[key] = {
#                     "title":    item["title"],
#                     "image":    item["image"],
#                     "brand":    state["parsed_intent"].get(
#                                     "brand", ""
#                                 ),
#                     "category": state["parsed_intent"].get(
#                                     "category", ""
#                                 ),
#                     "rating":   item["rating"],
#                     "reviews":  item["reviews"],
#                     "product_id": item.get("product_id"),
#                     "immersive_token": item.get("immersive_token"),
#                     "google_shopping_url": item.get("google_shopping_url"),
#                     "prices":   []
#                 }

#             # Update product_id, immersive_token, and google_shopping_url if not already set
#             if not grouped[key].get("product_id") and item.get("product_id"):
#                 grouped[key]["product_id"] = item["product_id"]
#             if not grouped[key].get("immersive_token") and item.get("immersive_token"):
#                 grouped[key]["immersive_token"] = item["immersive_token"]
#             if not grouped[key].get("google_shopping_url") and item.get("google_shopping_url"):
#                 grouped[key]["google_shopping_url"] = item["google_shopping_url"]

#             # Avoid duplicate platform entries
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

#             # Update image if current one is empty
#             if not grouped[key]["image"] and item["image"]:
#                 grouped[key]["image"] = item["image"]

#             # Update rating if current is 0
#             if grouped[key]["rating"] == 0 and item["rating"] > 0:
#                 grouped[key]["rating"] = item["rating"]

#         # Find lowest price per product & tag it
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

#             # Sort prices: lowest first
#             product["prices"] = sorted(
#                 product["prices"],
#                 key=lambda x: x["price"] if x["price"] > 0 else 999999
#             )

#             product["lowest_price"] = min_price
#             final.append(product)

#         # Sort all products by lowest price
#         final = sorted(
#             final,
#             key=lambda x: x.get("lowest_price", 999999)
#         )

#         # Resolve direct merchant links and prices for the top 40 products in the backend concurrently
#         top_to_resolve = final[:40]
#         with ThreadPoolExecutor(max_workers=40) as executor:
#             resolved_top = list(executor.map(resolve_single_product_offers, top_to_resolve))
        
#         # Merge resolved products back into final list
#         final = resolved_top + final[40:]

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
#     """
#     Applies user-selected filters on aggregated results.
#     Handles: platform, budget, min_budget, rating, sort.
#     """
#     try:
#         results = state.get("final_results", [])
#         filters = state.get("filters") or {}
#         intent  = state.get("parsed_intent", {})

#         # ── Platform filter ──
#         platform = filters.get("platform", "all")
#         if platform and platform != "all":
#             results = [
#                 r for r in results
#                 if any(
#                     p["platform"].lower() == platform.lower()
#                     for p in r.get("prices", [])
#                 )
#             ]

#         # ── Budget filter ──
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

#         # ── Rating filter ──
#         min_rating = filters.get("min_rating")
#         if min_rating:
#             results = [
#                 r for r in results
#                 if r.get("rating", 0) >= float(min_rating)
#             ]

#         # ── Sort ──
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
#     """
#     Takes filtered results.
#     Applies pagination.
#     Returns final clean response.
#     """
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
#     """
#     Catches any agent failure.
#     Logs the error.
#     Returns clean empty response instead of crashing.
#     """
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