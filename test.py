import json
from agent.graph import agent


def test(query: str):
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print('='*50)

    result = agent.invoke({
        "query":                    query,
        "parsed_intent":            {},
        "raw_results":              [],
        "final_results":            [],
        "filtered_results":         [],
        "filters":                  {},
        "page":                     1,
        "limit":                    100,
        "total_count":              0,
        "served_from_cache":        False,
        "execution_time":           None,
        "total_platforms_searched": 0,
        "platforms_failed":         [],
        "fallback_used":            False,
        "retry_count":              0,
        "suggestions":              [],
        "error":                    None
    })

    # ── Check for errors ──
    if result.get("error"):
        print(f"❌ ERROR: {result['error']}")
        return

    # ── Print parsed intent ──
    print("\n📌 Parsed Intent:")
    print(json.dumps(result["parsed_intent"], indent=2))

    # ── Print metadata ──
    print(f"\n⚡ Execution time:    {result.get('execution_time')}s")
    print(f"🏬 Platforms found:   {result.get('total_platforms_searched')}")
    print(f"❌ Platforms failed:  {result.get('platforms_failed')}")
    print(f"🔁 Fallback used:     {result.get('fallback_used')}")
    print(f"📦 Total results:     {result.get('total_count')}")

    # ── Print ALL products ──
    products = result.get("final_results", [])
    print(f"\n🛍️  All Products ({len(products)}):")
    print("-" * 50)

    for i, p in enumerate(products):   # ← removed [:3] limit
        print(f"\n  [{i+1}] {p['title']}")
        print(f"       Rating:  {p.get('rating')} ★  |  "
              f"Reviews: {p.get('reviews', 0)}")

        prices = p.get("prices", [])
        if prices:
            for price in prices:
                tag = " ← LOWEST" if price["is_lowest"] else ""
                disc = (f"  ({price['discount']}% off)"
                        if price.get("discount") else "")
                print(f"       {price['platform']:20}"
                      f" ₹{price['price']}{disc}{tag}")
        else:
            print("       No prices found")

    print("\n" + "-" * 50)
    print(f"✅ Showing {len(products)} of "
          f"{result.get('total_count')} total results")


# ─── Interactive Mode ─────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  PriceHunt — Test Mode")
    print("="*50)
    print("Type your search query and press Enter.")
    print("Type 'quit' to stop.\n")

    while True:
        try:
            query = input("🔍 Search: ").strip()

            if not query:
                print("⚠️  Please enter a search query.\n")
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Bye!")
                break

            print(f"\n⏳ Searching for '{query}'...")
            test(query)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break