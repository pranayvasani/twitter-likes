#!/usr/bin/env python3
import time, json, sqlite3, sys, random
from pathlib import Path
import requests

# Always use absolute path inside container
DATA_DIR = Path("/app/data")
TOKENS_FILE = DATA_DIR / "tokens.json"
DB_PATH = DATA_DIR / "likes.db"
PAGINATION_STATE = DATA_DIR / "pagination_state.json"

PAGE_SIZE = 100
MAX_RETRIES = 6
BACKOFF_BASE = 2.0

def load_tokens():
    if not TOKENS_FILE.exists():
        raise SystemExit("tokens.json not found. Run the OAuth flow first.")
    return json.loads(TOKENS_FILE.read_text())

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS likes (
            tweet_id TEXT PRIMARY KEY,
            author_id TEXT,
            author_username TEXT,
            text TEXT,
            created_at TEXT,
            raw_json TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT,
            name TEXT,
            profile_image_url TEXT,
            raw_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_state(state):
    DATA_DIR.mkdir(exist_ok=True)
    PAGINATION_STATE.write_text(json.dumps(state))

def load_state():
    if PAGINATION_STATE.exists():
        return json.loads(PAGINATION_STATE.read_text())
    return {}

def save_tweets_and_users(tweets, includes):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    author_map = {}
    if includes and "users" in includes:
        for u in includes["users"]:
            author_map[u["id"]] = u.get("username")
            cur.execute("""
                INSERT OR REPLACE INTO users
                (id, username, name, profile_image_url, raw_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                u.get("id"),
                u.get("username"),
                u.get("name"),
                u.get("profile_image_url"),
                json.dumps(u)
            ))

    for t in tweets:
        cur.execute("""
            INSERT OR IGNORE INTO likes
            (tweet_id, author_id, author_username, text, created_at, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            t.get("id"),
            t.get("author_id"),
            author_map.get(t.get("author_id")),
            t.get("text"),
            t.get("created_at"),
            json.dumps(t)
        ))

    conn.commit()
    conn.close()

def get_user_id(access_token: str) -> str:
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get("https://api.x.com/2/users/me", headers=headers, timeout=15)
    if resp.status_code != 200:
        raise SystemExit(f"Failed to get user info: {resp.status_code} {resp.text}")
    return resp.json()["data"]["id"]

def fetch_all_likes(access_token: str, user_id: str):
    headers = {"Authorization": f"Bearer {access_token}"}

    state = load_state()
    next_token = state.get("next_token")
    total = 0
    page = 0

    while True:
        page += 1
        params = {
            "max_results": PAGE_SIZE,
            "tweet.fields": "id,text,created_at,author_id,public_metrics,entities,lang",
            "expansions": "author_id",
            "user.fields": "id,username,name,profile_image_url",
        }
        if next_token:
            params["pagination_token"] = next_token

        attempt = 0
        consecutive_429 = 0
        MAX_BACKOFF = 900           # cap backoff to 15 minutes
        LONG_RETRY_THRESHOLD = 5    # after 5 consecutive 429s, do a long sleep
        LONG_SLEEP = 15 * 60        # 15 minutes

        while True:
            attempt += 1
            try:
                resp = requests.get(
                    f"https://api.x.com/2/users/{user_id}/liked_tweets",
                    headers=headers,
                    params=params,
                    timeout=30
                )
            except Exception as exc:
                wait = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
                jitter = random.uniform(0.5, 1.5)
                wait = wait * jitter
                print(f"Network error: {exc}. Sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                if attempt >= MAX_RETRIES:
                    raise SystemExit("Max network retries reached")
                continue

            status = resp.status_code
            if status == 200:
                consecutive_429 = 0
                break

            if status == 429:
                consecutive_429 += 1
                ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                if ra:
                    try:
                        wait = int(float(ra))
                    except Exception:
                        wait = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
                else:
                    wait = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
                    wait = wait * random.uniform(0.8, 1.3)

                print(f"Rate limited (429). Retry-After: {ra!s}. Sleeping {wait:.1f}s (attempt {attempt}).")
                if consecutive_429 >= LONG_RETRY_THRESHOLD:
                    print(f"Consecutive 429s reached {consecutive_429}. Sleeping long {LONG_SLEEP}s.")
                    time.sleep(LONG_SLEEP)
                    attempt = 0
                    consecutive_429 = 0
                else:
                    time.sleep(wait)
                if attempt >= MAX_RETRIES * 3:
                    raise SystemExit("Too many 429 retries, aborting.")
                continue

            if status in (500, 502, 503, 504):
                wait = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
                jitter = random.uniform(0.8, 1.2)
                wait = wait * jitter
                print(f"Server error {status}. Sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                if attempt >= MAX_RETRIES:
                    raise SystemExit("Max retries reached for server errors")
                continue

            body_preview = resp.text[:1000]
            print(f"Unexpected HTTP {status}: {body_preview}")
            try:
                DATA_DIR.mkdir(exist_ok=True)
                (DATA_DIR / "fetch_last_error.json").write_text(json.dumps({
                    "status": status,
                    "headers": dict(resp.headers),
                    "body": body_preview
                }))
            except Exception:
                pass
            raise SystemExit(f"Unexpected status {status}: aborting")

        j = resp.json()
        tweets = j.get("data", [])
        includes = j.get("includes", {})

        save_tweets_and_users(tweets, includes)
        total += len(tweets)

        meta = j.get("meta", {})
        next_token = meta.get("next_token")
        save_state({"next_token": next_token})

        print(f"Page {page}: {len(tweets)} tweets, total {total}")

        if not next_token:
            if PAGINATION_STATE.exists():
                PAGINATION_STATE.unlink()
            break

        time.sleep(0.3)

    print(f"Done. Total fetched: {total}")

def main():
    print("Initializing DB...")
    init_db()
    print("Loading tokens...")
    tokens = load_tokens()
    access_token = tokens.get("access_token")
    if not access_token:
        raise SystemExit("No access_token in tokens.json")

    print("Resolving user id...")
    user_id = get_user_id(access_token)
    print(f"User id = {user_id}")

    fetch_all_likes(access_token, user_id)

if __name__ == "__main__":
    main()
