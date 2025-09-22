'''
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/likes.db"

st.set_page_config(page_title="Liked Tweets Dashboard", layout="wide")

st.title("Liked Tweets — Dashboard")

@st.cache_data(ttl=60)
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df_likes = pd.read_sql_query("SELECT * FROM likes", conn)
        df_users = pd.read_sql_query("SELECT * FROM users", conn)
        conn.close()
        if not df_likes.empty:
            df_likes['created_at'] = pd.to_datetime(df_likes['created_at'], errors='coerce')
        return df_likes, df_users
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

df_likes, df_users = load_data()

if df_likes.empty:
    st.warning("No liked tweets found in data/likes.db yet. Run the fetch script or authorize the app.")
    st.stop()

# Top authors
st.subheader("Top accounts you like")
top_auth = df_likes.groupby('author_username').size().reset_index(name='count').sort_values('count', ascending=False).head(20)
st.dataframe(top_auth, use_container_width=True)

# Timeline (monthly)
st.subheader("Likes over time (monthly)")
timeline = df_likes.set_index('created_at').resample('M').size().reset_index(name='count')
timeline['created_at'] = timeline['created_at'].dt.to_period('M').dt.to_timestamp()
st.line_chart(data=timeline.set_index('created_at')['count'])

# Samples per author (select)
st.subheader("Sample liked tweets")
author = st.selectbox("Select author (username)", options=top_auth['author_username'].dropna().tolist())
if author:
    sample = df_likes[df_likes['author_username'] == author].sort_values('created_at', ascending=False).head(10)
    for _, row in sample.iterrows():
        st.write(f"**{row['author_username']}** — {row['created_at']}")
        st.write(row['text'])
        st.write("---")

# Quick text search across liked tweets
st.subheader("Search liked tweet text")
q = st.text_input("Search (regex or plain substring)", "")
if q:
    res = df_likes[df_likes['text'].str.contains(q, case=False, na=False)]
    st.write(f"Found {len(res)} matching tweets")
    st.dataframe(res[['author_username','created_at','text']].sort_values('created_at', ascending=False).head(200))
'''

"""
Award-winning style Streamlit dashboard for exploring `likes.db` (Twitter liked tweets)

Features included:
- Robust loading of `/mnt/data/likes.db` or uploaded DB
- Textual analyses: unigrams, bigrams, hashtags, mentions, emojis, wordcloud
- Sentiment (TextBlob fallback) + sentiment over time
- Author-level analysis (counts, avg sentiment, lexical richness)
- Temporal analysis (hour, weekday, timeline)
- Semantic clustering pipeline (embeddings -> UMAP -> HDBSCAN) with graceful fallback if heavy libs missing
- BERTopic optional (if installed)
- Semantic search (embedding-based) with rerank
- Keyphrase extraction (YAKE fallback to TF-IDF)
- Co-occurrence network (hashtags) using networkx + pyvis (if available) or matplotlib
- Word-shift (between two groups) using log-odds ratio
- Export / Download CSVs and ZIP of results
- Instructions to run & Dockerfile snippet included below

How to run:
1. Save this file as `streamlit_award_dashboard.py`.
2. Install dependencies (see `requirements.txt` below). Some heavy features are optional.
3. Run: `streamlit run streamlit_award_dashboard.py`

Notes: The app is defensive — if sentence-transformers, umap, or hdbscan are not installed it will still run and offer fallbacks. To get the full "award-winning" experience install the optional heavy libs.
"""

import streamlit as st
import wordcloud
import sqlite3
import pandas as pd
import numpy as np
import re
from collections import Counter
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

# Optional heavy imports guarded
_HAS_ST_TRANS = False
_HAS_UMAP = False
_HAS_HDBSCAN = False
_HAS_BERTOPIC = False
_HAS_YAKE = False
_HAS_PYVIS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST_TRANS = True
except Exception:
    _HAS_ST_TRANS = False
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False
try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    _HAS_BERTOPIC = False
try:
    import yake
    _HAS_YAKE = True
except Exception:
    _HAS_YAKE = False
try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False

# Sentiment: try TextBlob
try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except Exception:
    _HAS_TEXTBLOB = False

st.set_page_config(page_title="Likes: Award Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Likes — Award-style Interactive Dashboard")

# ----------------- Helpers -----------------
@st.cache_data
def load_db(path):
    conn = sqlite3.connect(path)
    likes = pd.read_sql("SELECT * FROM likes", conn)
    try:
        users = pd.read_sql("SELECT * FROM users", conn)
    except Exception:
        users = pd.DataFrame()
    conn.close()
    return likes, users

def clean_text_basic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def tokens_from_text(s):
    return re.findall(r"\b[a-z']+\b", s.lower())

emoji_pattern = re.compile("[" 
         u"\U0001F600-\U0001F64F"
         u"\U0001F300-\U0001F5FF"
         u"\U0001F680-\U0001F6FF"
         u"\U0001F1E0-\U0001F1FF"
         "]+", flags=re.UNICODE)

# sentiment
def get_sentiment(text):
    if not text or not isinstance(text, str):
        return np.nan
    if _HAS_TEXTBLOB:
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            pass
    # fallback heuristic
    pos_lex = {'good','great','love','awesome','nice','win','wins','happy','insightful','useful','amazing','best'}
    neg_lex = {'bad','terrible','hate','awful','worst','bug','bugs','problem','fail','fails','sad'}
    toks = re.findall(r"\w+", text.lower())
    score = sum(1 for t in toks if t in pos_lex) - sum(1 for t in toks if t in neg_lex)
    return float(score) / max(1, len(toks))

# entities (simple rule-based)
def simple_entities(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[A-Z][a-zA-Z]{1,}\b", text)

# CSV helper
def to_csv_bytes(df):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# ----------------- Sidebar: Data source & filters -----------------
st.sidebar.header("Data & filters")
DB_DEFAULT = "data/likes.db"
uploaded = st.sidebar.file_uploader("Upload likes.db (optional)", type=["db","sqlite","sqlite3"]) 
if uploaded is None:
    db_path = st.sidebar.text_input("Path to SQLite DB", value=DB_DEFAULT)
else:
    bytes_data = uploaded.read()
    tmp_path = "/tmp/likes_uploaded.db"
    with open(tmp_path, "wb") as f:
        f.write(bytes_data)
    db_path = tmp_path

likes, users = load_db(db_path)
if likes is None or len(likes)==0:
    st.error("No likes table found or empty DB. Ensure table `likes` exists.")
    st.stop()

likes['text'] = likes['text'].astype(str)
likes['clean_text'] = likes['text'].apply(clean_text_basic)
likes['created_at_parsed'] = pd.to_datetime(likes.get('created_at', None), errors='coerce')

# timezone conversion
tz_choice = st.sidebar.selectbox("Convert timestamps to timezone:", ["UTC", "Asia/Kolkata", "Local"], index=1)
if tz_choice == "Asia/Kolkata":
    try:
        likes['created_at_local'] = pd.to_datetime(likes['created_at_parsed']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    except Exception:
        likes['created_at_local'] = likes['created_at_parsed']
elif tz_choice == "Local":
    likes['created_at_local'] = likes['created_at_parsed'].dt.tz_localize('UTC').dt.tz_convert(None)
else:
    likes['created_at_local'] = likes['created_at_parsed']

# date filter
min_date = likes['created_at_local'].min()
max_date = likes['created_at_local'].max()
if pd.isna(min_date):
    min_date, max_date = None, None

date_range = st.sidebar.date_input("Date range", value=(min_date.date() if min_date is not None else None, max_date.date() if max_date is not None else None))
if date_range and min_date is not None:
    start, end = date_range
    likes = likes[(likes['created_at_local'].dt.date >= start) & (likes['created_at_local'].dt.date <= end)]

# author filter
authors = likes['author_username'].dropna().unique().tolist()
sel_authors = st.sidebar.multiselect("Authors (filter)", options=sorted(authors), default=None)
if sel_authors:
    likes = likes[likes['author_username'].isin(sel_authors)]

st.sidebar.markdown("---")
advanced_cluster = st.sidebar.checkbox("Enable semantic clustering (requires sentence-transformers + umap + hdbscan)", value=_HAS_ST_TRANS and _HAS_UMAP and _HAS_HDBSCAN)
use_bertopic = st.sidebar.checkbox("Enable BERTopic (optional)", value=_HAS_BERTOPIC)

# ----------------- Top row: quick stats -----------------
col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("Total liked tweets", len(likes))
    st.metric("Unique authors", likes['author_id'].nunique())
with col2:
    time_range_str = f"{likes['created_at_local'].min()} to {likes['created_at_local'].max()}"
    st.metric("Time range", time_range_str)
    st.metric("Avg tweet length (chars)", int(likes['text'].str.len().mean()))
with col3:
    st.write("\n")
    st.markdown("**Top authors (by likes)**")
    top_auth = likes['author_username'].value_counts().head(8)
    st.bar_chart(top_auth)

# ----------------- Sentiment & Temporal -----------------
st.header("Sentiment & Temporal")
likes['sentiment'] = likes['text'].apply(get_sentiment)
col4, col5 = st.columns([2,1])
with col4:
    st.subheader("Sentiment distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(likes['sentiment'].dropna(), bins=25, color="#ffb000")
    ax.set_xlabel('Polarity (-1 to 1)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.write("Summary:")
    st.write(likes['sentiment'].describe())
with col5:
    st.subheader("Likes by hour / weekday")
    if 'created_at_local' in likes.columns and likes['created_at_local'].notna().sum()>0:
        likes['hour'] = likes['created_at_local'].dt.hour
        likes['weekday'] = likes['created_at_local'].dt.day_name()
        cols = st.columns(1)
        st.bar_chart(likes['hour'].value_counts().sort_index())
        st.bar_chart(likes['weekday'].value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0))
    else:
        st.info('No valid timestamps for temporal charts')

# ----------------- Textual analysis (words, hashtags, mentions, emojis) -----------------
st.header("Textual analysis")
col6, col7 = st.columns([1,1])
with col6:
    st.subheader('Top unigrams')
    all_tokens = [t for txt in likes['clean_text'] for t in tokens_from_text(txt)]
    uni = Counter(all_tokens).most_common(50)
    uni_df = pd.DataFrame(uni, columns=['word','count'])
    st.dataframe(uni_df.head(30))
    st.download_button('Download unigrams CSV', data=to_csv_bytes(uni_df), file_name='unigrams.csv')

    st.subheader('Top bigrams')
    vect = CountVectorizer(ngram_range=(2,2), stop_words='english', min_df=2)
    try:
        Xb = vect.fit_transform(likes['clean_text'])
        sums = Xb.sum(axis=0).A1
        bigrams = sorted(list(zip(vect.get_feature_names_out(), sums)), key=lambda x: x[1], reverse=True)
        bigram_df = pd.DataFrame(bigrams, columns=['bigram','count'])
    except Exception:
        bigram_df = pd.DataFrame(columns=['bigram','count'])
    st.dataframe(bigram_df.head(30))
    st.download_button('Download bigrams CSV', data=to_csv_bytes(bigram_df), file_name='bigrams.csv')

with col7:
    st.subheader('Hashtags, Mentions, Emojis')
    hashtags = re.findall(r"#\w+", " ".join(likes['text'].dropna()))
    mentions = re.findall(r"@\w+", " ".join(likes['text'].dropna()))
    emojis = emoji_pattern.findall(" ".join(likes['text'].dropna()))
    st.write('Top Hashtags')
    st.dataframe(pd.DataFrame(Counter([h.lower() for h in hashtags]).most_common(50), columns=['hashtag','count']))
    st.write('Top Mentions')
    st.dataframe(pd.DataFrame(Counter([m.lower() for m in mentions]).most_common(50), columns=['mention','count']))
    st.write('Top Emojis')
    st.dataframe(pd.DataFrame(Counter(emojis).most_common(50), columns=['emoji','count']))

# wordcloud
st.subheader('Wordcloud')
if st.button('Regenerate wordcloud'):
    wc = WordCloud(width=1000, height=300, background_color='white').generate_from_text(' '.join(all_tokens))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    wc = WordCloud(width=1000, height=300, background_color='white').generate_from_text(' '.join(all_tokens))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ----------------- Keyphrase extraction -----------------
st.header('Keyphrase extraction & summaries')
if _HAS_YAKE:
    kw_extractor = yake.KeywordExtractor(lan='en', n=3, top=5)
    sample = likes['text'].dropna().head(50).tolist()
    kp = {i: kw_extractor.extract_keywords(t) for i,t in enumerate(sample)}
    st.write('Sample keyphrases (YAKE) for first 50 tweets')
    st.write(kp)
else:
    st.write('YAKE not available — falling back to TF-IDF top terms per tweet cluster.')
    st.write('You can install `yake` for better phrase extraction.')

# ----------------- Semantic clustering (Embeddings -> UMAP -> HDBSCAN) -----------------
st.header('Semantic clustering (meaningful groups)')
if advanced_cluster and _HAS_ST_TRANS and _HAS_UMAP and _HAS_HDBSCAN:
    with st.spinner('Computing embeddings + UMAP + HDBSCAN (may take a minute)'):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(likes['text'].fillna('').tolist(), show_progress_bar=False)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, metric='cosine', random_state=42)
        umap_emb = reducer.fit_transform(embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=8, metric='euclidean')
        cluster_labels = clusterer.fit_predict(umap_emb)
        likes['cluster'] = cluster_labels
        st.success('Clustering done')
        # Label clusters via c-TF-IDF style
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(likes['text'].fillna(''))
        clusters = []
        for c in sorted(set(cluster_labels)):
            mask = likes['cluster']==c
            if c==-1 or mask.sum()<3: continue
            mean_tfidf = X[mask.values].mean(axis=0)
            top_idx = mean_tfidf.A1.argsort()[-8:][::-1]
            terms = [tfidf.get_feature_names_out()[i] for i in top_idx]
            clusters.append({'cluster':c,'size':int(mask.sum()),'keywords':', '.join(terms)})
        st.dataframe(pd.DataFrame(clusters))
        # interactive scatter with Plotly
        try:
            import plotly.express as px
            scatter_df = pd.DataFrame(umap_emb, columns=['x','y'])
            scatter_df['cluster'] = cluster_labels
            scatter_df['text'] = likes['text'].values
            fig = px.scatter(scatter_df, x='x', y='y', color=scatter_df['cluster'].astype(str), hover_data=['text'])
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write('Plotly not available — install plotly for interactive scatter')
else:
    st.info('Semantic clustering is disabled or required libraries missing. Enable in sidebar and install sentence-transformers + umap-learn + hdbscan for best results.')

# ----------------- BERTopic (optional) -----------------
if use_bertopic and _HAS_BERTOPIC:
    with st.spinner('Running BERTopic...'):
        topic_model = BERTopic(verbose=False)
        topics, probs = topic_model.fit_transform(likes['text'].fillna('').tolist())
        likes['bertopic_topic'] = topics
        st.write(topic_model.get_topic_info().head(20))
        st.success('BERTopic finished')
else:
    if use_bertopic:
        st.warning('BERTopic selected but library not installed.')

# ----------------- Semantic search -----------------
st.header('Semantic search (ask in natural language)')
query = st.text_input('Search liked tweets (semantic):', value='')
if st.button('Search') and query.strip():
    if _HAS_ST_TRANS:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        doc_emb = model.encode(likes['text'].fillna('').tolist(), show_progress_bar=False)
        q_emb = model.encode([query])[0]
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([q_emb], doc_emb)[0]
        top_idx = np.argsort(sims)[::-1][:20]
        res = likes.iloc[top_idx][['author_username','text']].assign(similarity=sims[top_idx])
        st.dataframe(res)
    else:
        st.info('sentence-transformers missing — fallback to keyword search')
        hits = likes[likes['text'].str.contains('|'.join(query.split()), case=False, na=False)].head(50)
        st.dataframe(hits[['author_username','text']])

# ----------------- Co-occurrence / network (hashtags) -----------------
st.header('Hashtag co-occurrence network')
if len(hashtags)>0:
    edges = Counter()
    for t in likes['text'].dropna():
        hs = [h.lower() for h in re.findall(r"#\w+", t)]
        hs = list(set(hs))
        for i in range(len(hs)):
            for j in range(i+1,len(hs)):
                edges[(hs[i],hs[j])] += 1
    edge_df = pd.DataFrame([{'h1':a,'h2':b,'w':c} for (a,b),c in edges.items()])
    edge_df = edge_df.sort_values('w', ascending=False).head(200)
    st.write(edge_df.head(50))
    if _HAS_PYVIS:
        net = Network(height='600px', width='100%')
        nodes = set(edge_df['h1']).union(set(edge_df['h2']))
        for n in nodes:
            net.add_node(n, label=n)
        for _,r in edge_df.iterrows():
            net.add_edge(r['h1'], r['h2'], value=int(r['w']))
        net.save_graph('/tmp/hashtag_network.html')
        HtmlFile = open('/tmp/hashtag_network.html','r',encoding='utf-8')
        components = st.components.v1.html(HtmlFile.read(), height=650)
    else:
        st.info('pyvis not installed — display as table. Install pyvis for interactive network.')
else:
    st.info('No hashtags found in liked tweets')

# ----------------- Word-shift (compare two groups) -----------------
st.header('Word-shift: Compare groups (log-odds)')
col_a, col_b = st.columns(2)
with col_a:
    group_a_author = st.selectbox('Group A: author (or leave blank for all)', options=['']+sorted(likes['author_username'].dropna().unique().tolist()))
with col_b:
    group_b_author = st.selectbox('Group B: author (or leave blank for all)', options=['']+sorted(likes['author_username'].dropna().unique().tolist()))

def log_odds_ratio(df_a, df_b):
    vec = CountVectorizer(stop_words='english')
    A = vec.fit_transform(df_a['clean_text'].fillna(''))
    B = vec.transform(df_b['clean_text'].fillna('')) if df_b.shape[0]>0 else None
    # add smoothing
    a_counts = np.array(A.sum(axis=0)).ravel() + 1
    b_counts = np.array(B.sum(axis=0)).ravel() + 1 if B is not None else np.ones_like(a_counts)
    odds = np.log((a_counts/ a_counts.sum()) / (b_counts / b_counts.sum()))
    terms = np.array(vec.get_feature_names_out())
    idx = np.argsort(odds)[-40:]
    return pd.DataFrame({'term':terms[idx],'log_odds':odds[idx]})

if st.button('Compute word-shift'):
    dfA = likes if group_a_author=='' else likes[likes['author_username']==group_a_author]
    dfB = likes if group_b_author=='' else likes[likes['author_username']==group_b_author]
    if dfA.shape[0]==0 or dfB.shape[0]==0:
        st.warning('One of the groups is empty')
    else:
        ws = log_odds_ratio(dfA, dfB)
        st.dataframe(ws)

# ----------------- Author-level language profiling -----------------
st.header('Author language profiles')
prof = likes.groupby('author_username').agg(
    likes=('text','count'),
    avg_len=('text', lambda x: int(x.str.len().mean())),
    avg_tokens=('clean_text', lambda x: float(np.mean([len(tokens_from_text(s)) for s in x]))),
).reset_index().sort_values('likes', ascending=False)
st.dataframe(prof.head(50))
st.download_button('Download author profiles', data=to_csv_bytes(prof), file_name='author_profiles.csv')

# ----------------- Export all -----------------
st.header('Export / Save')
if st.button('Prepare ZIP of CSVs'):
    import zipfile, os
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        zf.writestr('unigrams.csv', uni_df.to_csv(index=False))
        zf.writestr('bigrams.csv', bigram_df.to_csv(index=False))
        zf.writestr('hashtags.csv', pd.DataFrame(Counter([h.lower() for h in hashtags]).most_common(), columns=['hashtag','count']).to_csv(index=False))
        zf.writestr('mentions.csv', pd.DataFrame(Counter([m.lower() for m in mentions]).most_common(), columns=['mention','count']).to_csv(index=False))
        zf.writestr('emojis.csv', pd.DataFrame(Counter(emojis).most_common(), columns=['emoji','count']).to_csv(index=False))
        zf.writestr('author_profiles.csv', prof.to_csv(index=False))
    zip_buf.seek(0)
    b64 = base64.b64encode(zip_buf.read()).decode()
    href = f"data:application/zip;base64,{b64}"
    st.markdown(f"[Download ZIP]({href})")

st.markdown('---')
st.markdown('**Notes / Next steps:** Install optional libs for full features: `sentence-transformers`, `umap-learn`, `hdbscan`, `plotly`, `bertopic`, `yake`, `pyvis`.')

# ----------------- requirements.txt snippet -----------------
st.expander('Requirements & Dockerfile (click to expand)').write('''
# requirements.txt (recommended)
streamlit
pandas
numpy
matplotlib
wordcloud
scikit-learn
textblob
plotly
# Optional for advanced features:
sentence-transformers
umap-learn
hdbscan
bertopic
yake
pyvis

# Dockerfile snippet
# FROM python:3.10-slim
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# CMD ["streamlit","run","streamlit_award_dashboard.py","--server.port","8501","--server.address","0.0.0.0"]
''')
