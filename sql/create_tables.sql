-- HyBench schema
-- Run once; Docker init script executes this automatically on first start.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
    id          SERIAL PRIMARY KEY,
    category    VARCHAR(50)     NOT NULL,
    price       DECIMAL(10, 2)  NOT NULL,
    brand       VARCHAR(100)    NOT NULL,
    rating      DECIMAL(3, 2)   NOT NULL,
    description TEXT            NOT NULL,
    embedding   vector(384)
);

-- Relational indexes used by Strategy B (filter-first)
CREATE INDEX idx_products_category ON products (category);
CREATE INDEX idx_products_price    ON products (price);
CREATE INDEX idx_products_rating   ON products (rating);
CREATE INDEX idx_products_cat_price_rating
    ON products (category, price, rating);

-- Materialised view for quick selectivity estimation
CREATE OR REPLACE VIEW v_category_stats AS
SELECT
    category,
    COUNT(*)                            AS total,
    MIN(price)                          AS min_price,
    MAX(price)                          AS max_price,
    ROUND(AVG(price)::numeric, 2)       AS avg_price,
    ROUND(AVG(rating)::numeric, 3)      AS avg_rating
FROM products
GROUP BY category;
