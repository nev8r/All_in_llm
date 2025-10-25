-- ============================
-- Text2SQL 示例数据库初始化脚本
-- ============================

DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS orders;

-- 用户表
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    city TEXT
);

-- 订单表
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    product TEXT,
    amount REAL,
    date TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- 插入用户数据
INSERT INTO users (user_id, name, age, city) VALUES
(1, 'Alice', 23, 'Tokyo'),
(2, 'Bob', 30, 'Osaka'),
(3, 'Charlie', 27, 'Nagoya'),
(4, 'Diana', 35, 'Kyoto');

-- 插入订单数据
INSERT INTO orders (order_id, user_id, product, amount, date) VALUES
(1, 1, 'Laptop', 1200.00, '2024-11-01'),
(2, 1, 'Mouse', 20.00, '2024-11-05'),
(3, 2, 'Keyboard', 50.00, '2024-12-10'),
(4, 3, 'Monitor', 200.00, '2025-01-15'),
(5, 4, 'Tablet', 600.00, '2025-02-01');
