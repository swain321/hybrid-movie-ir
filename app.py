"""Streamlit UI: streamlit run app.py | Evaluation: python app.py --evaluate"""
import argparse
import json
import os
import re
import threading
import unicodedata
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Entirely fictional movies, people, and relevance judgments.
MOVIES = json.loads(r'''
[
 {"id":"m01","title":"Beyond the Clock","genre":"Science Fiction Adventure","year":2021,"director":"Mira Sol","cast":["Arin Vale","Nia Moon"],"plot":"Astronauts explore a wormhole near Saturn. Time dilation separates a pilot from her daughter as the crew travels through time to save a dying Earth."},
 {"id":"m02","title":"The Last Orbit","genre":"Science Fiction Drama","year":2020,"director":"Leo Venn","cast":["Kai Stone","Eva Reed"],"plot":"An astronaut stranded on Mars grows food and repairs a radio. Engineers on Earth organize a dangerous space rescue mission."},
 {"id":"m03","title":"Dream Circuit","genre":"Science Fiction Thriller","year":2022,"director":"Mira Sol","cast":["Arin Vale","Zoe Flint"],"plot":"A thief enters shared dreams to steal corporate secrets. Layers of simulated reality make memories unreliable and time behave strangely."},
 {"id":"m04","title":"Midnight Ledger","genre":"Crime Mystery","year":2019,"director":"Dara Finch","cast":["Ivo Lane","Nia Moon"],"plot":"A detective follows hidden bank accounts after a journalist vanishes. A corrupt mayor and a criminal syndicate conceal a murder in a rainy city."},
 {"id":"m05","title":"Summer at Platform Nine","genre":"Romance Drama","year":2023,"director":"Elin Moss","cast":["Eva Reed","Noah Lake"],"plot":"Two strangers meet on a delayed train and fall in love. Years later they reunite at the station and confront missed opportunities."},
 {"id":"m06","title":"The Kitchen Underdogs","genre":"Comedy Family","year":2018,"director":"Oren Pike","cast":["Lila Brook","Kai Stone"],"plot":"An inexperienced cook joins a failing neighborhood restaurant. An unlikely team enters a cooking competition to save their business through friendship."},
 {"id":"m07","title":"Whispers Beneath","genre":"Horror Mystery","year":2024,"director":"Dara Finch","cast":["Zoe Flint","Ivo Lane"],"plot":"A family moves into a haunted coastal house. A child hears voices under the floor as an ancient ghost reveals the town's dark secret."},
 {"id":"m08","title":"The Iron Finish","genre":"Sports Drama","year":2020,"director":"Tomas Wren","cast":["Noah Lake","Lila Brook"],"plot":"An injured runner returns to training with an unconventional coach. Discipline and resilience help the athlete compete in a national marathon."},
 {"id":"m09","title":"Forest of Lanterns","genre":"Animation Fantasy Family","year":2022,"director":"Elin Moss","cast":["Nia Moon","Oren Fern"],"plot":"A young girl and a talking fox travel through an enchanted forest. They protect magical creatures and restore a stolen light to their village."},
 {"id":"m10","title":"Tomorrow Again","genre":"Science Fiction Comedy","year":2024,"director":"Leo Venn","cast":["Kai Stone","Lila Brook"],"plot":"A scientist's broken time machine traps a small town in a repeating day. Each time loop offers another chance to repair friendships and prevent a disaster."}
]
''')

QRELS = [
    ("Beyond the Clock", {"m01"}),
    ("Mira Sol Arin Vale", {"m01", "m03"}),
    ("space exploration mind-bending time travel", {"m01"}),
    ("astronaut survival and rescue on another planet", {"m02"}),
    ("dreams simulated reality stolen secrets", {"m03"}),
    ("detective investigating corruption and murder", {"m04"}),
    ("strangers fall in love and reunite years later", {"m05"}),
    ("cooking competition restaurant friendship", {"m06"}),
    ("haunted house ghost voices", {"m07"}),
    ("athlete recovering from injury to race again", {"m08"}),
    ("animated magical forest talking animals", {"m09"}),
    ("funny scientist repeating the same day", {"m10"}),
    ("Mira Sol space mission time dilation", {"m01"}),
    ("time travel time loop", {"m01", "m10"}),
]
MODEL = os.getenv("MOVIE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STEMMER = PorterStemmer()
# Preserve negation, which can affect a movie request's meaning.
STOP_WORDS = ENGLISH_STOP_WORDS - {"no", "not", "never", "nor"}


def tokenize(text):
    """Normalize accents/case, tokenize, remove stop words, then stem."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [STEMMER.stem(t) for t in re.findall(r"[a-z0-9]+", text)
            if t not in STOP_WORDS]


def document(movie):
    return (f"{movie['title']}. Genre: {movie['genre']}. {movie['plot']} "
            f"Cast: {', '.join(movie['cast'])}. Director: {movie['director']}. "
            f"Released in {movie['year']}.")


def ranked_indices(scores, positive_only=False):
    """Stable ties follow dataset order; zero sparse scores are not matches."""
    return [int(i) for i in np.argsort(-scores, kind="stable")
            if not positive_only or scores[i] > 0]


class HybridMovieIR:
    def __init__(self, model_name=MODEL, encoder=None):
        self.movies = MOVIES
        self.texts = [document(m) for m in self.movies]
        tokens = [tokenize(t) for t in self.texts]
        self.counts = [Counter(t) for t in tokens]
        self.lengths = np.array([len(t) for t in tokens], dtype=float)
        self.average_length = self.lengths.mean()
        df = Counter(t for terms in self.counts for t in terms)
        # Positive BM25 IDF (Lucene-style variant).
        self.idf = {t: np.log1p((len(tokens) - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            norm="l2", sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(self.texts)
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name, device="cpu")
        self.encoder = encoder
        # Serialize encoder access because Streamlit shares cached resources.
        self.lock = threading.Lock()
        self.embeddings = self.encode(self.texts)

    def encode(self, texts):
        with self.lock:
            vectors = self.encoder.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def bm25(self, query, k1=1.5, b=0.75):
        scores = np.zeros(len(self.movies))
        length_norm = k1 * (1 - b + b * self.lengths / self.average_length)
        for term in set(tokenize(query)):
            if term in self.idf:
                tf = np.array([c.get(term, 0) for c in self.counts])
                scores += self.idf[term] * tf * (k1 + 1) / (tf + length_norm)
        return scores

    def search(self, query, k=5, mode="hybrid", sparse="bm25", rrf_c=60):
        if mode not in {"hybrid", "sparse", "dense"}:
            raise ValueError("mode must be hybrid, sparse, or dense")
        if sparse not in {"bm25", "tfidf"}:
            raise ValueError("sparse must be bm25 or tfidf")
        if not 1 <= k <= len(self.movies) or rrf_c <= 0:
            raise ValueError("k must be within corpus size; rrf_c must be positive")
        if not query.strip() or not re.search(r"\w", query):
            return []
        bm = self.bm25(query)
        tf = (self.tfidf @ self.vectorizer.transform([query]).T).toarray().ravel()
        dense = np.clip(self.embeddings @ self.encode([query])[0], -1, 1)
        lexical = bm if sparse == "bm25" else tf
        sr = {i: r for r, i in enumerate(ranked_indices(lexical, True), 1)}
        dr = {i: r for r, i in enumerate(ranked_indices(dense), 1)}
        # Missing sparse matches contribute zero instead of arbitrary tie ranks.
        sc = np.array([1 / (rrf_c + sr[i]) if i in sr else 0
                       for i in range(len(self.movies))])
        dc = np.array([1 / (rrf_c + dr[i]) for i in range(len(self.movies))])
        score = {"hybrid": sc + dc, "sparse": lexical, "dense": dense}[mode]
        order = ranked_indices(score, positive_only=(mode == "sparse"))[:k]
        return [dict(movie=self.movies[i], score=float(score[i]),
                     bm25=float(bm[i]), tfidf=float(tf[i]), cosine=float(dense[i]),
                     sparse_rank=sr.get(i), dense_rank=dr[i],
                     sparse_rrf=float(sc[i]), dense_rrf=float(dc[i])) for i in order]


def precision_at_k(ids, relevant, k):
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ids[:k]) & relevant) / k


def recall_at_k(ids, relevant, k):
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ids, relevant):
    return next((1 / rank for rank, mid in enumerate(ids, 1)
                 if mid in relevant), 0.0)


def evaluate(engine, k=3, sparse="bm25"):
    """Macro P/R at K; MRR uses full returned rankings, not just top K."""
    if not 1 <= k <= len(engine.movies):
        raise ValueError("Evaluation k must be within corpus size")
    known = {m["id"] for m in engine.movies}
    if any(not relevant or not relevant <= known for _, relevant in QRELS):
        raise ValueError("Qrels must contain known, nonempty relevant ID sets")
    summaries, details = [], []
    for mode in ("sparse", "dense", "hybrid"):
        rows = []
        for query, relevant in QRELS:
            results = engine.search(query, len(engine.movies), mode, sparse)
            ids = [r["movie"]["id"] for r in results]
            row = {"mode": mode, "query": query,
                   f"Precision@{k}": precision_at_k(ids, relevant, k),
                   f"Recall@{k}": recall_at_k(ids, relevant, k),
                   "RR": reciprocal_rank(ids, relevant)}
            rows.append(row)
        summaries.append({"mode": mode, "sparse_method": sparse,
                          f"Precision@{k}": float(np.mean([r[f"Precision@{k}"] for r in rows])),
                          f"Recall@{k}": float(np.mean([r[f"Recall@{k}"] for r in rows])),
                          "MRR": float(np.mean([r["RR"] for r in rows]))})
        details.extend(rows)
    return {"summary": summaries, "per_query": details}


def run_ui():
    import streamlit as st

    st.set_page_config(page_title="Movie IR Lab", page_icon="🎬", layout="wide")
    st.title("🎬 Hybrid Movie Search")
    st.caption("10 fictional movies • BM25 / TF-IDF • MiniLM • Reciprocal Rank Fusion")

    @st.cache_resource
    def load_engine(model_name):
        return HybridMovieIR(model_name)

    with st.sidebar:
        st.header("Search settings")
        mode = st.selectbox("Retrieval mode", ["hybrid", "sparse", "dense"])
        sparse = st.selectbox("Sparse method", ["bm25", "tfidf"])
        k = st.slider("Results / evaluation K", 1, len(MOVIES), 3)
        st.caption("RRF constant: 60. Raw scores are not relevance probabilities.")
    try:
        with st.spinner("Loading model and building indexes…"):
            engine = load_engine(MODEL)
    except Exception as exc:
        st.error("Could not load the embedding model. Check the installation and "
                 "internet connection, or set MOVIE_EMBEDDING_MODEL to a local model folder.")
        st.exception(exc)
        st.stop()

    search_tab, evaluation_tab, dataset_tab = st.tabs(["Search", "Evaluation", "Dataset"])
    with search_tab:
        query = st.text_input("Describe a movie", "space exploration mind-bending time travel",
                              max_chars=1000)
        results = engine.search(query, k, mode, sparse)
        if not results:
            st.info("Enter a query with searchable terms. Sparse mode requires keyword overlap.")
        for rank, result in enumerate(results, 1):
            movie = result["movie"]
            with st.container(border=True):
                st.subheader(f"{rank}. {movie['title']} ({movie['year']})")
                st.write(f"**Genre:** {movie['genre']} · **Director:** {movie['director']}")
                st.write("**Cast:** " + ", ".join(movie["cast"]))
                st.write(movie["plot"])
                label = "RRF" if mode == "hybrid" else (sparse if mode == "sparse" else "cosine")
                st.write(f"**Ranking score ({label}): {result['score']:.6f}")
                with st.expander("Similarity and rank breakdown"):
                    st.json({key: value for key, value in result.items() if key != "movie"})
    with evaluation_tab:
        st.write("Fixed illustrative relevance judgments; unlisted movies are treated as nonrelevant. "
                 "MRR uses full rankings. These toy results do not establish real-world quality.")
        if st.button("Evaluate all three retrieval modes"):
            with st.spinner("Evaluating…"):
                report = evaluate(engine, k, sparse)
            st.dataframe(report["summary"], hide_index=True)
            st.dataframe(report["per_query"], hide_index=True)
            st.download_button("Download evaluation JSON", json.dumps(report, indent=2),
                               "evaluation.json", "application/json")
    with dataset_tab:
        st.dataframe(MOVIES, hide_index=True)
        st.write("Ground-truth queries")
        st.json([{ "query": q, "relevant_ids": sorted(ids)} for q, ids in QRELS])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sparse", choices=["bm25", "tfidf"], default="bm25")
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate(HybridMovieIR(), args.k, args.sparse), indent=2))
    else:
        run_ui()
