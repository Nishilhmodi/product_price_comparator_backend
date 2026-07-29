# PriceHunt

PriceHunt is an AI-powered product price comparison tool built for the Indian e-commerce market. It lets users search for any product in plain English or Hinglish and instantly compare prices across Amazon, Flipkart, Myntra, and Ajio — all in one place.

The backend uses a LangGraph AI agent that understands search intent, fetches live results via Google Shopping, groups identical products listed across multiple platforms, and returns a clean comparison with the lowest price highlighted.


## Features

- Natural language and Hinglish search support (e.g. "titan ghadi under 3k for dad")
- Live price comparison across Amazon, Flipkart, Myntra, and Ajio
- AI-powered search suggestions as you type
- Filters by platform, price range, and minimum rating
- Sort by price (low to high / high to low) or rating
- Lowest price badge on each product card
- Separate pages for Home, Results, and How It Works


## Tech Stack

**Backend**

- Python 3.10+
- FastAPI — REST API framework
- LangGraph — AI agent state machine
- LangChain Google GenAI — Gemini 2.5 Flash for intent parsing and suggestions
- SerpApi — Google Shopping results
- Redis — caching layer (optional)

**Frontend**

- React 19 with Vite 8
- React Router DOM 7
- Tailwind CSS 4
- Axios


## Project Structure

```
Product_price_comparator/
├── backend/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine wiring
│   │   ├── nodes.py          # All agent node functions
│   │   ├── state.py          # Agent state definition
│   │   └── tools.py
│   ├── api/
│   │   ├── models.py         # Pydantic request/response models
│   │   └── routes.py         # FastAPI route handlers
│   ├── scrapers/
│   │   ├── platforms.py      # Platform parsers and filters
│   │   └── serpapi.py        # SerpApi Google Shopping client
│   ├── cache.py              # Redis cache helpers
│   ├── config.py             # Environment variable loader
│   ├── main.py               # FastAPI app entry point
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   ├── ResultsPage.jsx
│   │   │   └── HowItWorksPage.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
└── .env
```


## Prerequisites

Make sure you have the following installed before proceeding:

- Python 3.10 or higher
- Node.js 18 or higher
- npm
- A SerpApi account with an API key — https://serpapi.com
- A Google AI Studio account with a Gemini API key — https://aistudio.google.com
- Redis (optional — only needed if you want caching)


## Getting API Keys

**SerpApi Key**

1. Sign up at https://serpapi.com
2. Go to your dashboard and copy the API key shown under "Your Private API Key"
3. Free plan includes 100 searches per month

**Gemini API Key**

1. Go to https://aistudio.google.com
2. Click "Get API Key" and create a new key
3. Copy the generated key


## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Product_price_comparator.git
cd Product_price_comparator
```

### 2. Configure environment variables

Create a `.env` file in the root of the project:

```bash
touch .env
```

Open it and add the following:

```
SERPAPI_KEY=your_serpapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
REDIS_URL=redis://localhost:6379
```

Replace the values with your actual API keys. If you are not using Redis, you can leave REDIS_URL as-is or remove it — the app runs fine without it.


### 3. Set up the backend

```bash
cd backend
```

Create and activate a virtual environment:

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive API docs are available at `http://localhost:8000/docs`.


### 4. Set up the frontend

Open a new terminal window, then:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.


## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| GET | /api/search | Search and compare products |
| GET | /api/suggest | AI-powered search suggestions |
| GET | /api/platforms | List of supported platforms |

**Search endpoint parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| q | string | Yes | Search query |
| platform | string | No | Filter by platform: all, amazon, flipkart, myntra, ajio |
| sort | string | No | Sort order: price_asc, price_desc, rating |
| max_price | integer | No | Maximum price filter |
| min_price | integer | No | Minimum price filter |
| min_rating | float | No | Minimum rating filter |
| page | integer | No | Page number (default: 1) |
| limit | integer | No | Results per page (default: 100, max: 200) |

**Example request:**

```
GET http://localhost:8000/api/search?q=titan+watch+under+3000&sort=price_asc&platform=all
```


## How the AI Agent Works

The backend uses a LangGraph state machine with the following pipeline:

```
START → parallel_intent_and_search → aggregator → filter → formatter → END
                    |
                 (retry, max 2 times if no results found)
                    |
                 error → END
```

1. The agent receives the raw user query
2. Gemini parses the intent (brand, budget, category, gender, etc.) while SerpApi fetches live Google Shopping results — both run simultaneously
3. Results are grouped by product using a bag-of-words semantic key so the same item listed on multiple platforms appears as one card
4. Filters and sorting are applied based on the user's query and any explicit filter parameters
5. The response is paginated and returned


## Environment Notes

- The backend must be running on port 8000 for the frontend to connect to it correctly
- Both servers must run at the same time during development
- The `.env` file must be placed in the root `Product_price_comparator/` directory, not inside `backend/` or `frontend/`
- Redis is optional. If Redis is unavailable, the app falls back to live search on every request with no errors


## Building for Production

To build the frontend for production:

```bash
cd frontend
npm run build
```

This generates a `dist/` folder. You can then serve it via any static hosting service (Vercel, Netlify, etc.) or configure FastAPI to serve it directly.

For the backend in production, replace `--reload` with a production-grade server setup using gunicorn:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```


## Common Issues

**Search returns no results**

- Check that your SERPAPI_KEY is valid and has remaining credits
- SerpApi free plan is limited to 100 searches per month

**Gemini suggestions not working**

- Verify your GEMINI_API_KEY is correct in the `.env` file
- Check that the Gemini API is enabled in your Google AI Studio account

**CORS errors in the browser**

- Make sure the backend is running on port 8000
- The backend allows all origins by default in development so this should not occur unless the backend is unreachable

**Frontend shows blank page**

- Ensure you have run `npm install` inside the `frontend/` directory
- Check the browser console for errors


## License

This project is open source and available under the MIT License.
