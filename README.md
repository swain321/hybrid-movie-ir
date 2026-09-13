# Hybrid Movie Information Retrieval System

Python 3.11 recommended. Ten fictional movies and 14 illustrative query judgments
are embedded in app.py. No database, API key, or NLTK corpus download is needed.
The pretrained MiniLM model requires internet on first launch and is cached locally.
The model weights are not bundled. CPU inference is selected for portability.

## Setup

Create and activate an environment:

    python -m venv .venv

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Windows Command Prompt:

    .venv\Scripts\activate.bat

macOS / Linux:

    source .venv/bin/activate

Install and launch:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Visit http://localhost:8501. Run commands from the extracted project directory.

## Evaluation and logic tests

    python app.py --evaluate --k 3 --sparse bm25
    python app.py --evaluate --k 3 --sparse tfidf
    python app.py --evaluate --k 3 > evaluation.json
    python -m unittest -v

Precision@K divides relevant hits by K even if fewer results are returned.
Recall@K divides hits by the query's number of relevant movies.
MRR averages the reciprocal rank of the first relevant result over all queries,
using the complete returned ranking. A missing relevant result contributes zero.
Summary Precision and Recall are macro averages across queries. No labels are
used for embedding generation or indexing. The toy judgments are illustrative,
not an independent benchmark; add held-out human judgments for real evaluation.

## Architecture

Sparse: accent/case normalization -> tokenization -> stop-word removal -> Porter
stemming -> TF-IDF cosine or BM25 with positive IDF and length normalization.
Dense: original metadata and plot text -> all-MiniLM-L6-v2 normalized embeddings
-> exact cosine search. Both use the same combined metadata/plot document.
Hybrid: RRF(d) = 1/(60 + sparse_rank(d)) + 1/(60 + dense_rank(d)).
A document absent from the sparse list contributes zero on that side. Full
rankings are fused before selecting K. Ties follow dataset order. Dense search
returns nearest neighbors even for queries outside the dataset's subject matter.
RRF is a ranking score, not a probability. BM25 and cosine have different scales.
TF-IDF is an alternative sparse branch, not an additional third fusion vote.
All comparison modes calculate the score breakdown, so the app always loads MiniLM.
Cached resources avoid rebuilding the model and index on each UI interaction.
Model encode calls are serialized for safe access to the shared resource.

## Examples

- Beyond the Clock
- space exploration mind-bending time travel
- Mira Sol space mission time dilation

## Local model support

Set MOVIE_EMBEDDING_MODEL to a directory containing a previously downloaded
SentenceTransformer model to run without internet after dependency installation.
For example, in PowerShell:

    $env:MOVIE_EMBEDDING_MODEL = 'C:\models\all-MiniLM-L6-v2'

## Validation scope

The included six offline tests exercise preprocessing, sparse exact-title
ranking, metric arithmetic, missing terms, RRF, and evaluation structure using a
test encoder. They do not validate actual MiniLM embeddings or browser rendering.
The real model and interactive UI were not executed in the generation environment.
Dependencies use bounded compatible ranges; this is not a fully locked environment.
