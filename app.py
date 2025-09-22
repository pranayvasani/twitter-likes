import os
import json
import sqlite3
from flask import Flask, request, redirect, url_for
from dotenv import load_dotenv
import tweepy
import traceback
import pathlib
import requests
from urllib.parse import urlencode
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
load_dotenv()

CLIENT_ID = os.getenv('TW_CLIENT_ID')
CLIENT_SECRET = os.getenv('TW_CLIENT_SECRET')
REDIRECT_URI = os.getenv('TW_REDIRECT_URI', 'http://localhost:5000/callback')
SCOPES = ["users.read", "tweet.read", "like.read", "offline.access"]
DB_PATH = 'likes.db'
TOKENS_FILE = 'tokens.json'

app = Flask(__name__)
app.config['OAUTH_HANDLER'] = None

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS likes (
        tweet_id TEXT PRIMARY KEY,
        author_id TEXT,
        author_username TEXT,
        text TEXT,
        created_at TEXT,
        raw_json TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return (
        '<h3>Twitter Likes Fetcher</h3>'
        '<p><a href="/connect">Connect with Twitter (start OAuth2 PKCE)</a></p>'
        '<p>After authorization the app will fetch your likes and save to <code>likes.db</code>.</p>'
    )

@app.route('/connect')
def connect():
    # create an OAuth2UserHandler via Tweepy which handles PKCE
    oauth2_user_handler = tweepy.OAuth2UserHandler(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
    )
    auth_url = oauth2_user_handler.get_authorization_url()
    # store handler for callback
    app.config['OAUTH_HANDLER'] = oauth2_user_handler
    return redirect(auth_url)

@app.route('/callback')
def callback():
    try:
        oauth2_user_handler = app.config.get('OAUTH_HANDLER')
        if oauth2_user_handler is None:
            return 'No OAuth handler found. Start at /connect', 400

        # ensure data dir
        pathlib.Path('data').mkdir(exist_ok=True)

        # grab code/state from query
        code = request.args.get('code')
        state = request.args.get('state')
        if not code:
            return 'Missing code in callback', 400

        # Use tweepy handler to exchange the full redirect URL for tokens
        # (requests_oauthlib expects the full redirect URL)
        token = oauth2_user_handler.fetch_token(request.url)

        # Save tokens for later (fetch_all script will reuse)
        with open('data/tokens.json', 'w') as f:
            json.dump(token, f, indent=2)

        # token should contain access_token (OAuth2 user token)
        access_token = token.get('access_token')
        if not access_token:
            return 'Token exchange did not return access_token. See data/tokens.json', 500

        headers = {"Authorization": f"Bearer {access_token}"}

        # 1) get the authenticated user's id
        me_resp = requests.get("https://api.x.com/2/users/me", headers=headers, timeout=15)
        if me_resp.status_code != 200:
            # save response for debugging
            with open('data/last_me_resp.json', 'w') as f:
                json.dump({"status": me_resp.status_code, "text": me_resp.text}, f, indent=2)
            return f"Failed to get user info: {me_resp.status_code}. Check data/last_me_resp.json", 500

        user_id = me_resp.json()['data']['id']

        # 2) fetch a small page of liked tweets (first page) with expansions (usernames)
        params = {
            "max_results": 50,
            "tweet.fields": "id,text,created_at,author_id,public_metrics,entities,lang",
            "expansions": "author_id",
            "user.fields": "id,username,name,profile_image_url"
        }
        likes_resp = requests.get(f"https://api.x.com/2/users/{user_id}/liked_tweets", headers=headers, params=params, timeout=30)
        # save raw response for debugging
        with open('data/last_likes_resp.json', 'w') as f:
            json.dump({"status": likes_resp.status_code, "body": likes_resp.json() if likes_resp.headers.get('content-type','').startswith('application/json') else likes_resp.text}, f, indent=2)

        if likes_resp.status_code not in (200, 201):
            return f"Failed to fetch liked tweets: {likes_resp.status_code}. Check data/last_likes_resp.json", 500

        resp_json = likes_resp.json()
        tweets = resp_json.get('data', [])
        includes = resp_json.get('includes', {})

        # persist into SQLite (data/likes.db)
        db_path = "data/likes.db"
        conn = sqlite3.connect(db_path)
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
        # build author map from includes
        author_map = {}
        if includes and 'users' in includes:
            for u in includes['users']:
                author_map[u['id']] = u.get('username')
                cur.execute("""
                    INSERT OR REPLACE INTO users (id, username, name, profile_image_url, raw_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (u.get('id'), u.get('username'), u.get('name'), u.get('profile_image_url'), json.dumps(u)))

        for t in tweets:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO likes (tweet_id, author_id, author_username, text, created_at, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (t.get('id'), t.get('author_id'), author_map.get(t.get('author_id')), t.get('text'), t.get('created_at'), json.dumps(t)))
            except Exception as e:
                # swallow and continue
                print("DB insert error:", e)

        conn.commit()
        conn.close()

        return f"Success — exchanged token and fetched {len(tweets)} liked tweets. Tokens saved to data/tokens.json; see data/likes.db."
    except Exception as e:
        tb = traceback.format_exc()
        pathlib.Path('data').mkdir(exist_ok=True)
        with open('data/error_trace.txt', 'w') as f:
            f.write(tb)
        return f"<pre>{tb}</pre>", 500


def save_likes_to_db(likes, includes=None):
    # build author_id -> username map from includes if available
    author_map = {}
    if includes and 'users' in includes:
        for u in includes['users']:
            author_map[u['id']] = u.get('username')

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for t in likes:
        tweet_id = t.get('id')
        author_id = t.get('author_id')
        username = author_map.get(author_id)
        text = t.get('text')
        created_at = t.get('created_at')
        raw = json.dumps(t)
        try:
            cur.execute('''INSERT OR REPLACE INTO likes (tweet_id, author_id, author_username, text, created_at, raw_json) VALUES (?, ?, ?, ?, ?, ?)''',
                        (tweet_id, author_id, username, text, created_at, raw))
        except Exception as e:
            print('DB save error', e)
    conn.commit()
    conn.close()

@app.route('/likes')
def likes():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM likes')
    count = cur.fetchone()[0]
    conn.close()
    return f"Total likes saved: {count}\n"

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    app.run(host=host, port=port)
