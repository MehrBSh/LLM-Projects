# ============================================================
# Hybrid RAG for MID: Medicines Information Dataset (MID.xlsx) (Fully Local Version)
# ------------------------------------------------------------
# - Chat backend: Ollama (Mistral)
# - Embeddings: Ollama bge-m3 (local, free)
# - Vector DB: ChromaDB (local)
# - SQL DB: SQLite (local)
# ============================================================

import os
import sqlite3
import pandas as pd
import textwrap
import json
import re
import time

from ollama import Client as OllamaClient
from chromadb import PersistentClient


# ============================================================
# CONFIGURATION
# ============================================================


MID_PATH = "path/to/your/MID.xlsx"
DB_PATH = "mid.db"
TABLE_NAME = "mid_drugs"

CHROMA_DIR = "./chroma_mid_db"
CHROMA_COLLECTION = "mid_vectors"

MODEL_CHAT = "mistral"
MODEL_EMBED = "bge-m3"

ollama = OllamaClient(host="http://localhost:11434")


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "name",
    "link",
    "contains",
    "productintroduction",
    "productuses",
    "productbenefits",
    "sideeffect",
    "howtouse",
    "howworks",
    "quicktips",
    "safetyadvice",
    "chemical_class",
    "habit_forming",
    "therapeutic_class",
    "action_class",
]


# ============================================================
# LOAD MID → SQLITE
# ============================================================

def load_mid_to_sqlite():
    print("Loading MID.xlsx into SQLite...")

    df = pd.read_excel(MID_PATH)

    # Normalize column names
    df.columns = [
        c.strip().lower().replace(" ", "").replace("_", "")
        for c in df.columns
    ]

    # Map normalized names to expected names
    col_map = {}
    for col in df.columns:
        for expected in EXPECTED_COLUMNS:
            if col == expected.replace("_", ""):
                col_map[col] = expected

    df = df.rename(columns=col_map)
    df = df[[c for c in EXPECTED_COLUMNS if c in df.columns]]

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print("SQLite DB ready.")
    return df


# ============================================================
# SQL HELPERS
# ============================================================

def extract_sql(text: str) -> str:
    """Extract SELECT query from LLM output."""
    idx = text.lower().find("select")
    if idx == -1:
        return "SELECT * FROM mid_drugs LIMIT 10;"
    sql = text[idx:].strip()

    # Safety: prevent hallucinated joins or invalid columns
    if " join " in sql.lower():
        return "SELECT * FROM mid_drugs LIMIT 10;"
    if "drug_name" in sql.lower():
        return "SELECT * FROM mid_drugs WHERE name LIKE '%keyword%' LIMIT 10;"

    return sql


def run_sql(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# ============================================================
# CHROMA VECTOR STORE
# ============================================================

def get_collection():
    client = PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except:
        return client.create_collection(
            CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )


def embed(text: str):
    resp = ollama.embeddings(model=MODEL_EMBED, prompt=text)
    return resp["embedding"]


def build_vector_store(df: pd.DataFrame):
    print("Building vector store...")

    col = get_collection()

    # Clear old data
    try:
        col.delete()
    except:
        pass

    docs = []
    ids = []
    metas = []

    for i, row in df.iterrows():
        combined = " ".join([
            str(row.get("name", "")),
            str(row.get("productuses", "")),
            str(row.get("howworks", "")),
        ])

        chunk = combined[:300].strip()
        if not chunk:
            continue

        docs.append(chunk)
        ids.append(str(i))
        metas.append({"name": row.get("name", "")})

    embeddings = [embed(d) for d in docs]

    col.add(
        documents=docs,
        ids=ids,
        embeddings=embeddings,
        metadatas=metas
    )

    print("Vector store built.")


def vector_search(query: str):
    col = get_collection()
    qvec = embed(query)
    res = col.query(query_embeddings=[qvec], n_results=5)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]


# ============================================================
# CHAT (MISTRAL)
# ============================================================

def chat(messages):
    prompt = ""
    for m in messages:
        prompt += f"{m['role']}: {m['content']}\n"
    prompt += "assistant:"

    resp = ollama.generate(model=MODEL_CHAT, prompt=prompt)
    return resp["response"]


# ============================================================
# HYBRID RAG
# ============================================================

SQL_SYSTEM = textwrap.dedent("""
You write SQL for a SQLite table named mid_drugs.

VALID COLUMNS:
name, link, contains, productintroduction, productuses, productbenefits,
sideeffect, howtouse, howworks, quicktips, safetyadvice, chemical_class,
habit_forming, therapeutic_class, action_class

RULES:
- Only use SELECT
- Never use JOIN
- Never invent columns
- If unsure, use:
  SELECT * FROM mid_drugs WHERE name LIKE '%keyword%' LIMIT 10;
""").strip()

ANSWER_SYSTEM = textwrap.dedent("""
You answer using ONLY:
- SQL rows
- vector documents

Never invent facts.
""").strip()


def generate_sql(question: str) -> str:
    messages = [
        {"role": "system", "content": SQL_SYSTEM},
        {"role": "user", "content": question},
    ]
    raw = chat(messages)
    return extract_sql(raw)


def answer_hybrid(question: str, df_sql, vec_docs):
    ctx = {
        "sql_rows": df_sql.to_dict(orient="records"),
        "vector_docs": vec_docs
    }

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "assistant", "content": json.dumps(ctx, indent=2)},
        {"role": "user", "content": question},
    ]
    return chat(messages)


def ask_mid(question: str):
    print("\nUser:", question)

    sql = generate_sql(question)
    print("SQL:", sql)

    try:
        df = run_sql(sql)
    except Exception as e:
        return f"SQL error: {e}"

    print("SQL rows:", len(df))

    vec = vector_search(question)
    print("Vector matches:", len(vec))

    return answer_hybrid(question, df, vec)


# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":
    # Load or build SQLite
    if not os.path.exists(DB_PATH):
        df_mid = load_mid_to_sqlite()
    else:
        conn = sqlite3.connect(DB_PATH)
        df_mid = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        print("Loaded MID from SQLite.")

    # Build vector store if missing
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        build_vector_store(df_mid.head(5000))
    else:
        print("Vector store already exists — skipping embedding.")

    print("\nHybrid RAG ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        print("\nAssistant:", ask_mid(q))
        print("\n" + "=" * 60 + "\n")
