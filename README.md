# Anime Recommender

A retrieval-augmented generation (RAG) application that recommends anime titles from a
natural-language description of a user's taste. Semantically similar titles are retrieved
from a local Chroma vector store built over anime synopses, and a Groq-hosted large
language model turns the retrieved context into three structured recommendations.

The retrieval chain is composed with LangChain Expression Language (LCEL) and runs on
LangChain 1.x.

---

## Table of contents

- [Architecture](#architecture)
- [How a request flows](#how-a-request-flows)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Installation](#installation)
- [Building the vector store](#building-the-vector-store)
- [Running the application](#running-the-application)
- [Module reference](#module-reference)
- [Data](#data)
- [Docker](#docker)
- [Kubernetes](#kubernetes)
- [Logging](#logging)
- [Error handling](#error-handling)
- [Dependency pinning](#dependency-pinning)
- [Troubleshooting](#troubleshooting)
- [Known issues and limitations](#known-issues-and-limitations)
- [License](#license)

---

## Architecture

| Concern | Implementation |
| --- | --- |
| User interface | Streamlit |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, executed locally |
| Vector store | Chroma, persisted to disk at `chroma_db/` |
| Retrieval | Similarity search, four documents per query |
| Generation | Groq chat completion, default model `qwen/qwen3.8-27b` |
| Chain composition | LCEL runnables from `langchain-core` |

The embedding model runs on the local machine through `sentence-transformers`. No embedding
API calls are made and no Hugging Face credentials are required. The only outbound network
dependency at query time is the Groq chat completion endpoint.

## How a request flows

1. The user submits a free-text query in the Streamlit text input.
2. `AnimeRecommendationPipeline.recommend` forwards the query to `AnimeRecommender`.
3. The LCEL chain fans out: the retriever embeds the query and returns the four most
   similar documents, while the raw query is passed through unchanged.
4. The retrieved documents are concatenated into a single `context` string, separated by
   blank lines.
5. `context` and `question` are rendered into the prompt template.
6. The prompt is sent to the Groq model at `temperature=0`.
7. The model response is parsed to a string and rendered as Markdown in the browser.

The chain retains the retrieved documents alongside the generated answer, so the sources
behind a recommendation can be inspected. See
[`src/recommender.py`](#srcrecommenderpy).

## Project layout

```
AnimeRecommender/
├── app/
│   ├── __init__.py
│   └── app.py                  Streamlit entry point
├── config/
│   ├── __init__.py
│   └── config.py               Environment configuration and validation
├── data/
│   ├── anime_with_synopsis.csv Source dataset
│   └── anime_updated.csv       Generated: single combined_info column
├── pipeline/
│   ├── __init__.py
│   ├── build_pipeline.py       Offline job: build and persist the vector store
│   └── pipeline.py             Runtime pipeline wiring
├── src/
│   ├── __init__.py
│   ├── data_loader.py          CSV preprocessing
│   ├── prompt_template.py      Prompt definition
│   ├── recommender.py          LCEL retrieval chain
│   └── vector_store.py         Chroma build and load
├── utils/
│   ├── __init__.py
│   ├── cutom_exception.py      Custom exception type
│   └── logger.py               Logging configuration
├── chroma_db/                  Persisted vector store, committed to the repository
├── Dockerfile
├── llmops-k8s.yaml             Kubernetes Deployment and Service
├── requirements.txt
└── setup.py
```

## Requirements

- Python 3.10
- A Groq API key, obtainable at https://console.groq.com
- Approximately 90 MB of disk space for the embedding model, which is downloaded to the
  Hugging Face cache on first run

## Configuration

Configuration is read from a `.env` file in the project root. This file is excluded by
`.gitignore` and must never be committed.

Create it with the following contents:

```
GROQ_API_KEY=your_groq_api_key_here
```

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Yes | none | Authenticates requests to the Groq API |
| `MODEL_NAME` | No | `qwen/qwen3.8-27b` | Overrides the Groq chat model |

`config/config.py` raises a `RuntimeError` at import time if `GROQ_API_KEY` is absent. This
is deliberate: the application fails immediately with an actionable message rather than
surfacing an opaque authentication error from deep inside the Groq client.

To select a different model, set `MODEL_NAME` to any chat model available to your Groq
account. The set of available models varies by account. To enumerate yours, run the
following in a Python session with the virtual environment active:

```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(sorted(model.id for model in client.models.list().data))
```

## Installation

```bash
git clone https://github.com/WahbMohamed2/AnimeRecommender.git
cd AnimeRecommender

python -m venv venv
venv\Scripts\activate
```

On macOS and Linux, activate with `source venv/bin/activate` instead.

```bash
pip install -e .
```

Installing with `-e` runs `setup.py`, which reads `requirements.txt` and registers the
`app`, `config`, `pipeline`, `src`, and `utils` packages on the import path. This matters:
the modules use absolute imports such as `from src.recommender import AnimeRecommender`,
which resolve only because the project is installed as a package. A plain
`pip install -r requirements.txt` installs the dependencies but leaves these imports
broken.

## Building the vector store

The repository ships with a prebuilt store in `chroma_db/`, so this step is not required
for a first run. Rebuild it only after changing the dataset, the embedding model, or the
chunking configuration.

```bash
python pipeline/build_pipeline.py
```

The job performs the following steps:

1. Reads `data/anime_with_synopsis.csv`, skipping malformed lines.
2. Verifies that the `Name`, `Genres`, and `sypnopsis` columns are present, and raises a
   `ValueError` naming any that are missing.
3. Drops rows with nulls in those three columns.
4. Composes a single `combined_info` field per row in the form
   `Title : <name> .. Overview : <synopsis> Genre : <genres>`.
5. Writes that single column to `data/anime_updated.csv`.
6. Loads the generated CSV, splits it with a 1000-character splitter, embeds each document,
   and persists the result to `chroma_db/`.

Chroma 0.4 and later persist automatically, so no explicit `persist()` call is needed.

Rebuilding writes into the existing `chroma_db/` directory rather than replacing it. To
guarantee a clean rebuild, delete the directory first.

## Running the application

```bash
streamlit run app/app.py
```

The application is served at http://localhost:8501.

Run this command from the project root. `AnimeRecommendationPipeline` defaults to the
relative path `persist_dir="chroma_db"`, so launching from another working directory
produces an empty vector store and recommendations with no grounding.

The pipeline is constructed inside a function decorated with `@st.cache_resource`, so the
embedding model is loaded and the vector store opened once per process rather than on every
interaction.

## Module reference

### app/app.py

Streamlit entry point. Loads environment variables, configures the page, initialises the
cached pipeline, and renders the query input and results. Submitting a query displays a
spinner while the recommendation is generated.

### config/config.py

Reads `GROQ_API_KEY` and `MODEL_NAME` from the environment and validates that the API key
is present.

### pipeline/pipeline.py

`AnimeRecommendationPipeline` wires the components together. Its constructor loads the
persisted vector store, converts it to a retriever, and injects the retriever along with
the API key and model name into `AnimeRecommender`. The `recommend` method delegates to the
recommender and returns the generated text. Failures in either phase are logged and
re-raised as `CustomException`.

### pipeline/build_pipeline.py

Offline entry point that runs `AnimeDataLoader` followed by
`VectorStoreBuilder.build_and_save_vectorstore`. Intended to be executed directly, not
imported.

### src/data_loader.py

`AnimeDataLoader` reads the source CSV, validates and cleans the required columns, derives
the `combined_info` text field, and writes the processed single-column CSV.

### src/vector_store.py

`VectorStoreBuilder` owns both halves of the vector store lifecycle.
`build_and_save_vectorstore` loads the processed CSV, splits it, embeds it, and persists it.
`load_vector_store` reopens an existing store for querying. `csv_path` defaults to an empty
string so that read-only callers can omit it.

### src/prompt_template.py

`get_anime_prompt` returns a `PromptTemplate` over the input variables `context` and
`question`. The prompt instructs the model to return exactly three titles, each with a plot
summary of two to three sentences and an explanation of why it matches the request, and to
state honestly when it does not know rather than fabricating titles.

### src/recommender.py

`AnimeRecommender` builds the retrieval chain:

```python
self.chain = RunnableParallel(
    source_documents=self.retriever,
    question=RunnablePassthrough(),
).assign(
    answer=(
        {
            "context": lambda x: format_docs(x["source_documents"]),
            "question": lambda x: x["question"],
        }
        | self.prompt
        | self.llm
        | StrOutputParser()
    )
)
```

`RunnableParallel` runs retrieval and pass-through concurrently. `assign` then adds the
generated `answer` while preserving `source_documents`, which reproduces the behaviour of
the `return_source_documents=True` option on the legacy `RetrievalQA` chain.

Two accessors are exposed. `get_recommendation` returns the answer string alone and is what
the Streamlit application calls. `get_recommendation_with_sources` returns the full
dictionary containing `question`, `source_documents`, and `answer`, which is useful for
evaluation and debugging.

Note that `StrOutputParser` in LangChain 1.x returns a `TextAccessor`. This is a subclass of
`str` and behaves as a string everywhere, including in `st.markdown`.

### utils/logger.py

Configures the root logger to write to `logs/log_<YYYY-MM-DD>.log` at `INFO` level and
exposes `get_logger(name)`. The `logs/` directory is created on import if absent.

### utils/cutom_exception.py

`CustomException` wraps an originating exception together with a contextual message and
appends the file name and line number extracted from the active traceback.

## Data

`data/anime_with_synopsis.csv` contains 269 rows with the columns `MAL_ID`, `Name`,
`Score`, `Genres`, and `sypnopsis`. The misspelling of `sypnopsis` is present in the source
dataset and is therefore the name the code expects.

Only `Name`, `Genres`, and `sypnopsis` are used. `MAL_ID` and `Score` are carried in the
file but never read, so scores do not influence retrieval or ranking.

Every row survives cleaning, producing 269 documents in the vector store under the Chroma
collection named `langchain`.

## Docker

```bash
docker build -t llmops-app:latest .
docker run -p 8501:8501 --env-file .env llmops-app:latest
```

The image is based on `python:3.10-slim`, installs the project with `pip install -e .`,
exposes port 8501, and starts Streamlit bound to `0.0.0.0` in headless mode.

Pass credentials at run time with `--env-file`, as shown above. Do not bake them into the
image.

The build copies the prebuilt `chroma_db/` directory into the image and does not run
`build_pipeline.py`. The vector store must therefore be present in the build context, which
is why it is committed to this repository. If you choose to exclude it, add a
`RUN python pipeline/build_pipeline.py` step to the Dockerfile so the image builds its own
store.

There is no `.dockerignore` in this repository. Because the Dockerfile uses `COPY . .`, a
build performed in a working tree that contains a `.env` file copies that file into an image
layer, where it remains readable even if a later layer deletes it. Add a `.dockerignore`
listing at least `.env`, `venv/`, `__pycache__/`, and `logs/` before publishing any image.

## Kubernetes

`llmops-k8s.yaml` defines a single-replica `Deployment` and a `LoadBalancer` `Service` that
forwards port 80 to container port 8501.

The Deployment sources its environment from a Secret named `llmops-secrets` via `envFrom`,
so credentials are never stored in the manifest. Create the Secret before applying it:

```bash
kubectl create secret generic llmops-secrets --from-literal=GROQ_API_KEY=your_groq_api_key_here
kubectl apply -f llmops-k8s.yaml
```

`imagePullPolicy` is set to `IfNotPresent` and the image reference is the unqualified tag
`llmops-app:latest`, which assumes an image already present in the local image store of the
cluster node. This suits a local cluster such as Minikube or kind. For a remote cluster,
push the image to a registry and replace the reference with its fully qualified name.

## Logging

Logs are written to `logs/log_<YYYY-MM-DD>.log`, one file per day, at `INFO` level, in the
format `timestamp - level - message`. The pipeline logs initialisation, every received
query, successful generation, and all failures with their messages. The `logs/` directory is
excluded from version control.

## Error handling

Both pipeline phases follow the same discipline: catch the exception, log it at `ERROR`
level, and re-raise it wrapped in `CustomException` with a message describing the phase.
The wrapper preserves the original exception and records the file and line at which it
surfaced, so a stack trace identifies both the failure site and the stage of the pipeline.

Configuration is an intentional exception to this pattern. A missing `GROQ_API_KEY` raises
at import time, before any pipeline object is constructed.

## Dependency pinning

Every dependency in `requirements.txt` is constrained to a major version range.

This is not incidental. The project was originally written against LangChain 0.x with
unpinned requirements. A later reinstall resolved to LangChain 1.x, which removed the
`langchain.chains` and `langchain.prompts` modules outright, and the application stopped
starting. The retrieval chain has since been rewritten in LCEL against `langchain-core`,
and the ranges prevent a comparable major-version jump from silently breaking imports again.

`setup.py` strips comments and blank lines when it reads `requirements.txt`, so the file can
carry explanatory comments without producing invalid requirement specifiers.

`langchain-community` remains a dependency for a single import, `CSVLoader`, used only by
the offline build job. The package is being sunset upstream and emits a deprecation warning
on import. Replacing it means constructing `Document` objects from pandas directly, which
would alter the indexed text slightly by dropping the `combined_info: ` field prefix that
`CSVLoader` prepends, and would require rebuilding the store.

## Troubleshooting

**`RuntimeError: GROQ_API_KEY is not set`**

No `.env` file exists in the project root, or it does not define `GROQ_API_KEY`. Note that
the variable is `GROQ_API_KEY`, not `GROQ_KEY_API`.

**`ModuleNotFoundError: No module named 'src'`**

The project is not installed. Run `pip install -e .` from the project root.

**`ModuleNotFoundError: No module named 'langchain.chains'` or `'langchain.prompts'`**

Both modules were removed in LangChain 1.0. Their absence indicates code written for
LangChain 0.x. The chain in this repository does not import them; encountering this error
means an older revision is being run.

**Recommendations are generic or unrelated to the dataset**

The vector store is empty or was not found. Confirm that `streamlit run` was invoked from
the project root and that `chroma_db/` is present, then rebuild with
`python pipeline/build_pipeline.py`.

**A Groq error naming the model**

The configured model is unavailable to the account in use. Enumerate the accessible models
with the snippet in [Configuration](#configuration) and set `MODEL_NAME` accordingly.

**First run is slow**

The embedding model is downloaded to the Hugging Face cache on first use. Subsequent runs
load it from disk.

## Known issues and limitations

- The dataset contains 269 titles. Recommendations are drawn from this subset, not from
  anime as a whole, and the prompt permits the model to supplement retrieved context with
  its own knowledge, so suggested titles are not guaranteed to be present in the store.
- `utils/cutom_exception.py` carries a misspelled filename. It is referenced by that name in
  `pipeline/pipeline.py` and `pipeline/build_pipeline.py`. Renaming it requires updating
  both imports.
- `CustomException.get_detailed_error_message` reads `sys.exc_info()` rather than the
  exception passed to it. Raised outside an active `except` block, it reports
  `Unknown File` and `Unknown Line`.
- Retrieval depth is fixed at the default of four documents and is not configurable without
  editing `pipeline/pipeline.py`.
- `MAL_ID` and `Score` are present in the dataset but unused, so popularity and rating play
  no part in retrieval or ranking.
- There is no automated test suite.
- There is no `.dockerignore`. See [Docker](#docker).

## License

No license has been declared for this repository. All rights are reserved by the author
by default.
