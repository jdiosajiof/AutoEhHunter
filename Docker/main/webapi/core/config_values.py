from typing import Any

from .constants import CONFIG_SPECS


_HNSW_MAX_VECTOR = 2000
_HNSW_MAX_HALFVEC = 4000


def resolve_text_vec_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute effective vector type, stored dimension, and index type for text embeddings.

    Returns dict with keys: dim, stored_dim, storage, index, type_sql, cast_sql.
    """
    dim = max(64, int(cfg.get("EMB_TEXT_DIM") or 1024))
    matryoshka = as_bool(cfg.get("EMB_TEXT_MATRYOSHKA"), False)
    storage_pref = str(cfg.get("EMB_TEXT_STORAGE") or "auto").strip().lower()
    index_pref = str(cfg.get("EMB_TEXT_INDEX") or "hnsw").strip().lower()
    if index_pref not in ("hnsw", "ivfflat", "none"):
        index_pref = "hnsw"

    def _best_storage(d: int) -> str:
        if d <= _HNSW_MAX_VECTOR:
            return "vector"
        if d <= _HNSW_MAX_HALFVEC:
            return "halfvec"
        return "vector"

    def _max_indexed_dim(st: str) -> int:
        if st == "vector":
            return _HNSW_MAX_VECTOR
        if st == "halfvec":
            return _HNSW_MAX_HALFVEC
        return 0

    if storage_pref == "auto":
        if matryoshka:
            storage = _best_storage(dim)
            stored_dim = min(dim, _max_indexed_dim(storage)) if index_pref != "none" else dim
        else:
            storage = _best_storage(dim)
            stored_dim = dim
    else:
        storage = storage_pref if storage_pref in ("vector", "halfvec", "bit") else _best_storage(dim)
        if matryoshka:
            max_idx = _max_indexed_dim(storage)
            stored_dim = min(dim, max_idx) if (index_pref != "none" and max_idx > 0) else dim
        else:
            stored_dim = dim

    if index_pref != "none" and stored_dim > _max_indexed_dim(storage):
        index_pref = "none"

    if storage == "bit":
        type_sql = f"bit({stored_dim})"
        cast_sql = "::bit"
        ops_sql = "bit_hamming_ops"
    elif storage == "halfvec":
        type_sql = f"halfvec({stored_dim})"
        cast_sql = "::halfvec"
        ops_sql = "halfvec_cosine_ops"
    else:
        type_sql = f"vector({stored_dim})"
        cast_sql = "::vector"
        ops_sql = "vector_cosine_ops"

    return {
        "dim": dim,
        "stored_dim": stored_dim,
        "storage": storage,
        "index": index_pref,
        "type_sql": type_sql,
        "cast_sql": cast_sql,
        "ops_sql": ops_sql,
        "matryoshka": matryoshka,
    }


def str_bool(value: bool) -> str:
    return "1" if bool(value) else "0"


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def normalize_value(key: str, raw: Any) -> str:
    spec = CONFIG_SPECS.get(key, {"type": "text", "default": ""})
    kind = str(spec.get("type", "text"))
    default = spec.get("default", "")
    if kind == "bool":
        return str_bool(as_bool(raw, bool(default)))
    if kind == "int":
        try:
            number = int(str(raw).strip())
        except Exception:
            number = int(default)
        number = max(int(spec.get("min", number)), min(int(spec.get("max", number)), number))
        return str(number)
    if kind == "float":
        try:
            number = float(str(raw).strip())
        except Exception:
            number = float(default)
        number = max(float(spec.get("min", number)), min(float(spec.get("max", number)), number))
        return str(number)
    if raw is None:
        return str(default)
    return str(raw).strip()
