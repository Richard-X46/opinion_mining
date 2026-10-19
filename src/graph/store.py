import chromadb
import logging

from src.config.settings import CHROMA_COLLECTION

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Pin cosine space explicitly: distances come back as (1 - cos_sim),
# which node_retrieve converts back to similarities for source ranking.
collection = chroma_client.get_or_create_collection(
    CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
)

# get_or_create does NOT backfill metadata on a pre-existing collection:
# a collection created before the cosine pin defaults to L2 space, where
# distances are squared-Euclidean (>= 1 for most pairs) and the 1 - d
# conversion in node_retrieve clamps everything to 0.00. Self-heal by
# recreating (stored comments are a re-ingestible cache, not source data).
if (collection.metadata or {}).get("hnsw:space") != "cosine":
    logging.warning(
        "Chroma collection '%s' is in space %r, expected cosine — recreating.",
        CHROMA_COLLECTION, (collection.metadata or {}).get("hnsw:space"),
    )
    chroma_client.delete_collection(CHROMA_COLLECTION)
    collection = chroma_client.get_or_create_collection(
        CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def prune_other_sessions(keep_ids: list[str]) -> int:
    """Delete docs from sessions outside keep_ids, plus untagged legacy docs.

    Gives each new session a blank slate: retrieval is already session-scoped,
    this reclaims the disk. Stored comments are a re-ingestible cache, never
    source data, so dropping them is always safe. Returns docs removed.
    """
    keep = set(keep_ids)
    got = collection.get(include=["metadatas"], limit=100000)
    ids = got.get("ids", []) or []
    metas = got.get("metadatas", []) or []
    drop = [
        i for i, m in zip(ids, metas)
        if not (m or {}).get("session_id") or m["session_id"] not in keep
    ]
    if drop:
        collection.delete(ids=drop)
        logging.info(f"Chroma prune: removed {len(drop)} docs from stale/untagged sessions.")
    return len(drop)