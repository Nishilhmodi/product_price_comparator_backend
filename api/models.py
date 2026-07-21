from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from enum import Enum

# ─── Enums ────────────────────────────────────────────

class SortOption(str, Enum):
    price_asc  = "price_asc"
    price_desc = "price_desc"
    rating     = "rating"
    relevance  = "relevance"

class Platform(str, Enum):
    all      = "all"
    amazon   = "amazon"
    flipkart = "flipkart"
    myntra   = "myntra"
    ajio     = "ajio"
    nykaa    = "nykaa"

# ─── Request Models ───────────────────────────────────
# Shape of what frontend SENDS to backend

class SearchRequest(BaseModel):
    query       : str = Field(..., example="The Bear House shirts")
    platform    : Platform = Field(Platform.all, example="amazon")
    sort        : SortOption = Field(SortOption.relevance, example="price_asc")
    max_price   : Optional[int] = Field(None, example=1000)
    min_rating  : Optional[float] = Field(None, example=100)
    page        : int = Field(1, example=1)
    limit       : int = Field(20, example=20)

class SuggestRequest(BaseModel):
    query: str = Field(..., example="The Bear House shirts")
    limit: int = Field(5, example=5)

# ─── Response Models ──────────────────────────────────
# Shape of what backend RETURNS to frontend

class PlatformPrice(BaseModel):
    platform:  str = Field(..., example="amazon")
    price:     int = Field(..., example=999)
    url:       str = Field(..., example="https://www.amazon.in/dp/B0C5Y1Z1X1")
    is_lowest: bool = Field(..., example=True)
    discount:  Optional[float] = Field(None, example=10)  # percentage discount

class Product(BaseModel):
    title:       str = Field(..., example="The Bear House Shirts")
    image:       str = Field(..., example="https://example.com/image.jpg")
    rating:      float = Field(..., example=4.5)
    reviews:     int = Field(..., example=100)
    prices:      List[PlatformPrice]
    lowest_price: Optional[int] = Field(None, example=999)
    brand:       Optional[str]   = Field(None, example="The Bear House")
    category:    Optional[str]   = Field(None, example="Shirts")
    product_id:  Optional[str]   = Field(None, example="13354099892770877362")
    immersive_token: Optional[str] = Field(None, example="7OQnY3icbVPLruNEFER8AQs...")
    google_shopping_url: Optional[str] = Field(None, example="https://www.google.com/search?...")
    resolved:    Optional[bool]  = Field(False, example=True)

class SearchResponse(BaseModel):
    query:   str = Field(..., example="The Bear House shirts")
    total:   int = Field(..., example=100)
    page:    int = Field(..., example=1)
    results: List[Product]
    source:  str = Field(..., example="live", description="live or cache")

class SuggestResponse(BaseModel):
    query:       str = Field(..., example="The Bear House shirts")
    suggestions: List[str] = Field(..., example=["The Bear House shirts for men", "The Bear House shirts for women"])

class HealthResponse(BaseModel):
    status:     str = Field(..., example="ok")
    cache:      str = Field(..., example="connected", description="connected or disconnected")
    version:    str = Field(..., example="1.0.0")
    message:    Optional[str] = Field(None, example="All systems operational")

# ─── Agent Models ─────────────────────────────────────
# Internal models used by AI agent only

class ParsedIntent(BaseModel):
    clean_query:    str = Field(..., example="The Bear House shirts")
    brand:          Optional[str]  = Field(None, example="The Bear House")
    category:       Optional[str]  = Field(None, example="Shirts")
    budget:         Optional[int]  = Field(None, example=1000)
    min_budget:     Optional[int]  = Field(None, example=500)
    gender:         Optional[str]  = Field(None, example="Men", description="Man, Woman, Unisex")
    color:          Optional[str]  = Field(None, example="Blue")
    sort_intent:    Optional[str]  = Field(None, example="price_asc", description="price_asc, price_desc, rating, relevance")
    query_language: str            = Field(..., example="en", description="Language of the query, e.g., en, hi, guj, etc.")

class ErrorResponse(BaseModel):
    error:      str = Field(..., example="An error occurred", description="A brief description of the error")
    detail:     Optional[str] = Field(None, example="Detailed error message", description="A more detailed explanation of the error, if available")
    code:       Optional[str] = Field(None, example="ERROR_CODE", description="A specific error code for programmatic handling, if applicable")
    retry:      bool          = Field(False, example=False, description="Indicates whether the request can be retried")
    suggestion: Optional[str] = Field(None, example="Try a different search query", description="A suggestion for the user to resolve the error, if applicable")

class InsightsResponse(BaseModel):
    title:          str = Field(..., example="Timex Men's Analog Watch")
    truthful_score: int = Field(..., example=85)
    summary:        str = Field(..., example="A reliable everyday watch with a durable strap, but lacks water resistance for swimming.")
    pros:           List[str] = Field(..., example=["Durable build", "Classic versatile design", "Excellent value for money"])
    cons:           List[str] = Field(..., example=["Basic water resistance", "No chronograph functions", "Strap is slightly stiff at first"])
    recommendation: str = Field(..., example="Buy Now", description="Buy Now, Wait for Drop, or Avoid")
    source:         str = Field(..., example="live", description="live or cache")

class ProductOffersResponse(BaseModel):
    product_id: Optional[str] = None
    offers: List[PlatformPrice]
    source: str = "live"

