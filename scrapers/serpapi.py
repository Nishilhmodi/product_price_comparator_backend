import requests
from config import SERPAPI_KEY

ALLOWED_PLATFORMS = ["amazon", "flipkart", "myntra", "ajio"]


def search_products(query: str, num: int = 100):
    """
    Search Google Shopping via SerpApi.
    Returns products from allowed platforms only.
    num: number of results to request (max 100 on paid plans)
    """
    try:
        # Append allowed store filters to the query to maximize allowed platform results
        # in a single SerpApi token call (1 credit only).
        expanded_query = f"{query.strip()} (amazon OR flipkart OR myntra OR ajio)"
        
        params = {
            "engine":  "google_shopping",
            "q":       expanded_query,
            "gl":      "in",
            "hl":      "en",
            "num":     num,
            "api_key": SERPAPI_KEY
        }

        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()

        results = response.json().get("shopping_results", [])

        # Filter to allowed platforms only
        filtered = [
            r for r in results
            if any(
                platform in r.get("source", "").lower()
                for platform in ALLOWED_PLATFORMS
            )
        ]

        return filtered

    except requests.exceptions.Timeout:
        print("[SerpApi] Request timed out")
        return []

    except requests.exceptions.RequestException as e:
        print(f"[SerpApi] Request failed: {str(e)}")
        return []

    except Exception as e:
        print(f"[SerpApi] Unexpected error: {str(e)}")
        return []