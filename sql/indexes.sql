-- HyBench v0.1: HNSW index only.
-- IVFFlat deferred to v0.2.
-- Executed by benchmark scripts after data is loaded; not part of initial schema.

DROP INDEX IF EXISTS idx_products_hnsw;

-- HNSW index with cosine distance operator.
-- m=16 (connections per layer, default), ef_construction=64 (build-time candidate list).
-- ef_search is session-level: SET hnsw.ef_search = <value>;
CREATE INDEX idx_products_hnsw
    ON products
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});
