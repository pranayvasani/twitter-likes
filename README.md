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
