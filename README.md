# Toronto Smart City Planning Agent

An agent that provides spatial, context-aware urban planning insights for the City of Toronto. The tool maps geographic coordinates to Toronto's ward boundaries and surfaces the relevant urban profile metrics for three distinct user personas: city planners & developers.

## Overview

The app is a single entry point (`app.py`) that routes users to one of three role-specific dashboards:

- **Government** — for city planners working on policy and zoning decisions.
- **Constructor** — for developers evaluating sites and build conditions.
questions.

Under the hood, a `SpatialContextEngine` (in `core_engine.py`) loads Toronto ward GeoJSON boundaries and a precomputed urban profile dataset, builds an in-memory spatial index using Shapely, and resolves any `(lng, lat)` pair to the containing ward along with its associated profile metrics.

## Project Structure

```
toronto-smart-city-v2/
├── app.py                      # Streamlit landing page with persona selection
├── core_engine.py              # SpatialContextEngine — point-in-polygon ward lookup
├── debug_neighbourhoods.py     # Debug helper for neighbourhood data
├── debug_roads.py              # Debug helper for road data
├── requirements.txt
├── .streamlit/                 # Streamlit configuration
├── data/                       # GeoJSON boundaries and urban profile JSON
├── pages/
│   ├── 1_Government.py         # City planner dashboard
│   ├── 2_Constructor.py        # Developer dashboard
└── src/                        # Supporting modules/services/data pipeline
```

## Requirements

- Python 3.9+
- Dependencies (see `requirements.txt`):
  - `streamlit`
  - `flask`, `flask-socketio`, `flask-cors`
  - `openai`
  - `geopandas`, `shapely` (via geopandas)
  - `pandas`, `requests`
  - `python-dotenv`
  - `adm-zip`

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sushumnaspradeep/toronto-smart-city-v2.git
   cd toronto-smart-city-v2
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables. Create a `.env` file in the project root with your API keys (e.g. for the OpenAI-powered resident chat):

   ```env
   NGC_API_KEY=your_key_here
   ```

## Running the App

Launch the Streamlit app from the project root:

```bash
streamlit run app.py
```

The app opens in your browser. Pick a persona from the landing page to enter the corresponding dashboard.

## How It Works

The `SpatialContextEngine` does the heavy lifting:

1. Loads Toronto ward boundaries from a GeoJSON file in `data/`.
2. Loads precomputed urban profile metrics (one entry per ward) from a JSON file in `data/`.
3. Builds an in-memory spatial index pairing each ward polygon with its profile.
4. Exposes `get_context_by_coordinates(lng, lat)`, which runs a point-in-polygon check and returns the ward name, ward number, and urban profile for that location — or an `outside_boundaries` response if the coordinate falls outside the city.

This lets every page in the app ask a single, simple question — *"what do we know about this spot?"* — and get a consistent answer.

## Debug Scripts

`debug_neighbourhoods.py` and `debug_roads.py` are standalone helpers for inspecting the underlying geospatial datasets while developing or troubleshooting.

## Authors

Built by [sushumna](https://github.com/sushumnaspradeep), [monil](https;github.com/monmin-2), [amogh](https://github.com/amoghmanju)