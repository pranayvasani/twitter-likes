# Twitter Likes Explorer

A Dockerized Flask + Streamlit app to fetch and analyze your liked tweets.

## Setup

1. Clone this repo.
2. Copy `example.env` → `.env` and fill in with your Twitter/X Developer App credentials:
   - CLIENT_ID
   - CLIENT_SECRET
   - REDIRECT_URI = http://localhost:5000/callback

3. Run with Docker Compose:

```bash
docker compose up --build

Workflow

Visit http://localhost:5000
 and click Connect with Twitter.
This saves data/tokens.json and data/likes.db.

The fetcher service automatically starts fetching all liked tweets (resume-capable).
Data is stored in data/likes.db.

Visit http://localhost:8501
 for the dashboard.

Notes

If you see 429 Too Many Requests, the fetcher backs off and retries automatically.

All state is persisted in data/.

Safe to stop/restart: progress is resumed from data/pagination_state.json.
