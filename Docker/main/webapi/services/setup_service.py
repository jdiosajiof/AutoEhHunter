from __future__ import annotations

from typing import Any
from pathlib import Path

import psycopg
import requests

from ..core.config_values import resolve_text_vec_config
from .auth_service import build_dsn


def validate_db_connection(host: str, port: int, db: str, user: str, password: str, sslmode: str = "prefer") -> tuple[bool, str, str]:
    dsn = build_dsn(host=host, port=port, db=db, user=user, password=password, sslmode=sslmode)
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "ok", dsn
    except Exception as e:
        return False, str(e), dsn


def validate_lrr(base: str, api_key: str, timeout_s: int = 8) -> tuple[bool, str]:
    b = str(base or "").strip().rstrip("/")
    if not b:
        return True, "empty base (optional)"
    if not b.startswith("http://") and not b.startswith("https://"):
        b = f"http://{b}"
    url = f"{b}/api/info"
    headers: dict[str, Any] = {}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
        headers["X-API-Key"] = key
    try:
        r = requests.get(url, headers=headers, timeout=max(2, int(timeout_s)))
        if 200 <= int(r.status_code) < 300:
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, str(e)


def _build_text_vec_index_sql(index_name: str, table: str, column: str, vc: dict[str, Any]) -> str:
    idx = str(vc.get("index") or "none")
    ops = str(vc.get("ops_sql") or "vector_cosine_ops")
    if idx == "none":
        return f"-- index {index_name} skipped (EMB_TEXT_INDEX=none)"
    return f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} USING {idx} ({column} {ops});"


def _build_migration_sql(vc: dict[str, Any]) -> str:
    type_sql = str(vc.get("type_sql") or "vector(1024)")
    storage = str(vc.get("storage") or "vector")
    stored_dim = int(vc.get("stored_dim") or 1024)
    target_typname = storage if storage in ("vector", "halfvec", "bit") else "vector"
    _COL_INDEX = {
        ("works", "desc_embedding"): "idx_works_desc_vec",
        ("semantic_memory", "embedding"): "idx_semantic_memory_vec",
    }
    parts: list[str] = []
    for (table, column), idx_name in _COL_INDEX.items():
        parts.append(f"""
DO $$
DECLARE
    cur_typname text;
    cur_typmod  int;
BEGIN
    SELECT t.typname, a.atttypmod INTO cur_typname, cur_typmod
    FROM pg_attribute a
    JOIN pg_type t ON t.oid = a.atttypid
    WHERE a.attrelid = '{table}'::regclass AND a.attname = '{column}';
    IF cur_typname IS NOT NULL AND (cur_typname <> '{target_typname}' OR cur_typmod <> {stored_dim}) THEN
        EXECUTE 'DROP INDEX IF EXISTS {idx_name}';
        EXECUTE 'ALTER TABLE {table} ALTER COLUMN {column} TYPE {type_sql} USING NULL';
        RAISE NOTICE 'Migrated {table}.{column} from %(%) to {type_sql}', cur_typname, cur_typmod;
    END IF;
END $$;""")
    return "\n".join(parts)


def _substitute_schema_sql(sql: str, cfg: dict[str, Any]) -> str:
    vc = resolve_text_vec_config(cfg)
    type_sql = str(vc.get("type_sql") or "vector(1024)")

    sql = sql.replace("__TEXT_VEC_TYPE__", type_sql)

    sql = sql.replace(
        "__IDX_WORKS_DESC_VEC__",
        _build_text_vec_index_sql("idx_works_desc_vec", "works", "desc_embedding", vc),
    )
    sql = sql.replace(
        "__IDX_SEMANTIC_MEMORY_VEC__",
        _build_text_vec_index_sql("idx_semantic_memory_vec", "semantic_memory", "embedding", vc),
    )

    sql = sql.replace("__TEXT_VEC_MIGRATE__", _build_migration_sql(vc))

    return sql


def init_core_schema(dsn: str, schema_path: str = "", cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
    s = str(dsn or "").strip()
    if not s:
        return False, "missing dsn"

    candidates: list[Path] = []
    if str(schema_path or "").strip():
        candidates.append(Path(str(schema_path).strip()))
    candidates.extend(
        [
            Path("/app/textIngest/schema.sql"),
            Path(__file__).resolve().parents[2] / "textIngest" / "schema.sql",
        ]
    )

    schema_file = next((p for p in candidates if p.exists() and p.is_file()), None)
    if not schema_file:
        return False, "schema.sql not found"

    try:
        sql = schema_file.read_text(encoding="utf-8")
        effective_cfg = cfg or {}
        if not effective_cfg:
            try:
                from .config_service import resolve_config
                effective_cfg, _ = resolve_config()
            except Exception:
                pass
        sql = _substitute_schema_sql(sql, effective_cfg)
        with psycopg.connect(s, connect_timeout=15) as conn:
            conn.execute("SET statement_timeout = '5min'")
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        return True, "schema initialized"
    except Exception as e:
        return False, str(e)
