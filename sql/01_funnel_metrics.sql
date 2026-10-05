-- One row per session with nested boolean stages; no event-join duplication.
SELECT channel, is_mobile, COUNT(*) AS sessions,
       SUM(product_view) AS product_views, SUM(add_to_cart) AS carts,
       SUM(purchase) AS purchases,
       AVG(purchase * 1.0) AS purchase_rate,
       SUM(add_to_cart) * 1.0 / NULLIF(SUM(product_view), 0) AS product_to_cart_rate,
       SUM(purchase) * 1.0 / NULLIF(SUM(add_to_cart), 0) AS cart_to_purchase_rate,
       SUM(revenue_usd) AS revenue_usd
FROM sessions
GROUP BY channel, is_mobile;
