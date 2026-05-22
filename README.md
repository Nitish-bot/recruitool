# Recruitool — LinkedIn Profile Analyzer

An AI-powered recruitment assistant that scrapes LinkedIn profiles via Apify and lets you chat with candidate data using Google's Gemini.

## What it does

1. **Scrape** LinkedIn profiles from URLs using Apify's LinkedIn Profile Scraper.
2. **View** candidate profiles in a clean card-based UI (name, location, company, education, experience, recent posts).
3. **Chat with candidates** via Gemini — ask questions like *"Who has the most experience?"* or *"Compare their technical backgrounds."*

## Setup

**Prerequisites:** Python ≥ 3.10, an [Apify](https://apify.com/) account, and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
uv sync          # or: pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
APIFY_KEY = "your-apify-api-token"
GEMINI_API_KEY = "your-gemini-api-key"
```

## Usage

```bash
streamlit run app.py
```

1. Paste LinkedIn profile URLs in the sidebar (one per line).
2. Enter your Gemini API key if not already in secrets.
3. Click **Start Scraping** to send URLs to Apify.
4. If the job isn't instant, click **Check Status / Retry** to poll for results.
5. Once data loads, use the chat input to query candidates with natural language.
