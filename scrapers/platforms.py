from urllib.parse import urlparse, parse_qs
import re

ALLOWED_PLATFORMS = {
    "amazon", "flipkart", "myntra", "ajio"
}


def normalize_platform(name: str) -> str:
    """
    Normalize platform names from SerpApi.
    SerpApi returns inconsistent names like:
    "Amazon.in", "amazon.in", "Amazon" → all become "Amazon"
    """
    name = name.lower().strip()
    if "amazon"   in name: return "Amazon"
    if "flipkart" in name: return "Flipkart"
    if "myntra"   in name: return "Myntra"
    if "ajio"     in name: return "Ajio"
    return name.title()


def is_allowed_platform(source: str) -> bool:
    """
    Check if platform is in our allowed list.
    Only show Amazon, Flipkart, Myntra, Ajio.
    """
    source_lower = source.lower()
    return any(
        platform in source_lower
        for platform in ALLOWED_PLATFORMS
    )


def extract_destination_url(url: str) -> str:
    """
    Extract direct platform URL from a Google redirect URL if possible.
    """
    if not url:
        return ""
    if "google" not in url:
        return url
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        # Common parameters for redirects: 'url', 'q', 'adurl'
        for param in ["url", "q", "adurl"]:
            if param in params and params[param]:
                dest = params[param][0]
                if dest.startswith("http") and "google" not in dest:
                    return dest
    except Exception:
        pass
    return url


def extract_direct_link(item: dict) -> str:
    """
    Extract direct platform product link from SerpApi result.

    SerpApi returns Google redirect URLs like:
    https://www.google.com/search?ibp=oshop&q=...&prds=...

    Priority:
    1. Clean direct link from "link" or "product_link" (Google redirect bypassed)
    2. Google redirect link (which redirects directly to product page)
    3. Build search URL from platform + product title as last resort
    """
    link = item.get("link", "")
    product_link = item.get("product_link", "")

    # Priority 1 — Try to extract and clean direct links
    clean_link = extract_destination_url(link)
    if clean_link and "google.com" not in clean_link:
        return clean_link

    clean_prod_link = extract_destination_url(product_link)
    if clean_prod_link and "google.com" not in clean_prod_link:
        return clean_prod_link

    # Priority 2 — Fallback to Google redirect (takes user directly to product page)
    if link:
        return link
    if product_link:
        return product_link

    # Priority 3 — Build platform search URL
    source = item.get("source", "").lower()
    title  = item.get("title", "")
    query  = title.replace(" ", "+")

    if "amazon"   in source:
        return f"https://www.amazon.in/s?k={query}"

    if "flipkart" in source:
        return f"https://www.flipkart.com/search?q={query}"

    if "myntra"   in source:
        return f"https://www.myntra.com/{query.replace('+', '-')}"

    if "ajio"     in source:
        return f"https://www.ajio.com/search/?text={query}"

    return ""


def parse_discount(item: dict):
    """
    Extract discount from SerpApi response.
    SerpApi sometimes returns discount in:
    - direct "discount" field
    - inside "extensions" list as "15% off"
    """
    # Check direct discount field
    if item.get("discount"):
        try:
            return float(item.get("discount", 0))
        except:
            return None

    # Check inside extensions list
    # extensions: ["15% off", "Free delivery"]
    extensions = item.get("extensions", [])
    for ext in extensions:
        if "% off" in str(ext).lower():
            try:
                return float(
                    str(ext).lower()
                    .replace("% off", "")
                    .strip()
                )
            except:
                return None

    return None


def parse_price(price_str: str) -> int:
    """
    Convert price string to integer.
    "₹1,995.00" → 1995
    "₹3,24,675" → 324675 (valid high price)
    "1995"      → 1995
    """
    if not price_str:
        return 0
    try:
        # Remove currency symbols and commas
        # Keep only digits and decimal point
        cleaned = str(price_str)
        cleaned = cleaned.replace("₹", "").replace(",", "").strip()

        # Convert to float first then int
        # handles "1995.00" correctly
        price = int(float(cleaned))

        # Sanity check — reject unrealistic prices
        # Max realistic price for Indian e-commerce = ₹5,00,000
        if price > 500000:
            return 0

        return price

    except (ValueError, TypeError):
        return 0


def parse_reviews(reviews_val) -> int:
    """
    Convert reviews value to integer robustly.
    Handles float-like strings ("1.7", "1,200.5"), "1.7k", etc.
    """
    if not reviews_val:
        return 0
    try:
        cleaned = str(reviews_val).lower().replace(",", "").strip()
        # Handle "k" suffix for thousands (e.g., "1.7k" -> 1700)
        if "k" in cleaned:
            cleaned = cleaned.replace("k", "").strip()
            return int(float(cleaned) * 1000)
        # Handle "m" suffix for millions (e.g., "1.2m" -> 1200000)
        if "m" in cleaned:
            cleaned = cleaned.replace("m", "").strip()
            return int(float(cleaned) * 1000000)
        # Standard float/int conversion
        return int(float(cleaned))
    except (ValueError, TypeError):
        return 0


def parse_rating(rating_val) -> float:
    """
    Convert rating value to float robustly.
    Handles float numbers, strings like "4.5 out of 5", "4.2 stars", etc.
    """
    if not rating_val:
        return 0.0
    try:
        val_str = str(rating_val).strip()
        match = re.search(r'([0-9.]+)', val_str)
        if match:
            return float(match.group(1))
        return float(rating_val)
    except:
        return 0.0


def parse_result(item: dict) -> dict:
    """
    Parse a single SerpApi shopping result
    into a clean standardized dict.
    """
    return {
        "title":    item.get("title", ""),
        "price":    parse_price(item.get("price", "0")),
        "platform": normalize_platform(
                            item.get("source", "Unknown")
                        ),
        "image":    item.get("thumbnail", ""),
        "link":     extract_direct_link(item),
        "rating":   parse_rating(item.get("rating", 0)),
        "reviews":  parse_reviews(item.get("reviews", 0)),
        "discount": parse_discount(item),
        "product_id": item.get("product_id"),
        "immersive_token": item.get("immersive_product_page_token"),
        "google_shopping_url": item.get("product_link")
    }


def parse_and_filter_results(raw_results: list, query: str = "") -> list:
    """
    Parse all results, filter to allowed platforms only, and remove irrelevant accessories.
    """
    parsed = []
    query_lower = query.lower().strip()
    
    # ─── Query Relevance Filter ───
    # Split query into words, filter out stopwords, and ensure title matches at least one query keyword.
    stopwords = {"under", "above", "price", "best", "online", "for", "with", "show", "lowest", "gifting", "compare", "store", "deals", "in", "and", "or", "of", "to", "a", "the"}
    query_words = [
        w.lower().strip() for w in re.split(r'\s+', query_lower)
        if w.strip() and w.lower().strip() not in stopwords
    ]
    
    # Strict brand checks (ensure brand queries don't cross-match other brands)
    brand_map = {
        "iphone": ["iphone", "apple"],
        "apple": ["apple", "iphone", "ipad", "macbook"],
        "samsung": ["samsung", "galaxy"],
        "oneplus": ["oneplus", "nord"],
        "realme": ["realme"],
        "oppo": ["oppo"],
        "vivo": ["vivo"],
        "timex": ["timex"],
        "titan": ["titan"],
        "casio": ["casio"],
        "fastrack": ["fastrack"],
        "sony": ["sony"],
        "nike": ["nike"],
        "puma": ["puma"],
        "minimalist": ["minimalist"]
    }
    
    required_brand_keywords = []
    for brand_key, aliases in brand_map.items():
        if brand_key in query_lower:
            required_brand_keywords.extend(aliases)
            
    # Common accessory keywords to screen out if the user didn't explicitly search for them
    accessory_keywords = [
        "cover", "case", "tempered glass", "screen protector", "screen guard", 
        "back guard", "skin", "pouch", "sleeve", "charger", "cable", "adapter", 
        "temperedglass", "strap", "band"
    ]
    
    # We only screen out the accessory keyword if it's not present in the user's query
    active_filters = [
        kw for kw in accessory_keywords
        if kw not in query_lower
    ]

    for item in raw_results:
        source = item.get("source", "")

        # Skip if not an allowed platform
        if not is_allowed_platform(source):
            continue

        title = item.get("title", "")
        title_lower = title.lower()
        
        # Check query relevance (ensure title contains at least one keyword from the search query)
        if query_words:
            if not any(word in title_lower for word in query_words):
                continue
                
        # Check strict brand match if a brand is present in the query
        if required_brand_keywords:
            if not any(brand_word in title_lower for brand_word in required_brand_keywords):
                continue
        
        # Check if the title matches any filtered accessory keyword using word boundaries
        is_accessory = False
        for kw in active_filters:
            if " " in kw:
                if kw in title_lower:
                    is_accessory = True
                    break
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
                    is_accessory = True
                    break
                    
        if is_accessory:
            continue

        parsed.append(parse_result(item))

    return parsed