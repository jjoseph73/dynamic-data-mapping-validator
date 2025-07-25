-- =============================================================================
-- Source Database Schema (Legacy System)
-- File: sql/init_source.sql
-- =============================================================================

-- Create legacy schema
CREATE SCHEMA IF NOT EXISTS legacy;

-- Set search path
SET search_path TO legacy, public;

-- =============================================================================
-- CUSTOMERS TABLE (Legacy naming conventions)
-- =============================================================================
CREATE TABLE legacy.customers (
    cust_id SERIAL PRIMARY KEY,
    cust_name VARCHAR(100) NOT NULL,
    email_addr VARCHAR(255),
    phone_num VARCHAR(20),
    created_dt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status_cd CHAR(1) DEFAULT 'A' CHECK (status_cd IN ('A', 'I', 'S', 'D')),
    credit_limit DECIMAL(10,2),
    last_login_dt TIMESTAMP,
    updated_dt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add comments for documentation
COMMENT ON TABLE legacy.customers IS 'Legacy customer data with old naming conventions';
COMMENT ON COLUMN legacy.customers.cust_id IS 'Legacy customer ID (maps to customer_id in modern schema)';
COMMENT ON COLUMN legacy.customers.cust_name IS 'Customer full name (maps to full_name in modern schema)';
COMMENT ON COLUMN legacy.customers.email_addr IS 'Email address (maps to email in modern schema)';
COMMENT ON COLUMN legacy.customers.phone_num IS 'Phone number (maps to phone in modern schema)';
COMMENT ON COLUMN legacy.customers.status_cd IS 'Status code: A=Active, I=Inactive, S=Suspended, D=Deleted';
COMMENT ON COLUMN legacy.customers.created_dt IS 'Record creation timestamp';
COMMENT ON COLUMN legacy.customers.last_login_dt IS 'Last login timestamp';

-- =============================================================================
-- ORDERS TABLE (Legacy naming conventions)
-- =============================================================================
CREATE TABLE legacy.orders (
    order_id SERIAL PRIMARY KEY,
    cust_id INTEGER NOT NULL REFERENCES legacy.customers(cust_id) ON DELETE CASCADE,
    order_dt DATE NOT NULL,
    total_amt DECIMAL(10,2) NOT NULL CHECK (total_amt >= 0),
    order_status VARCHAR(20) DEFAULT 'PENDING' CHECK (order_status IN ('PENDING', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'CANCELLED')),
    ship_addr TEXT,
    created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add comments
COMMENT ON TABLE legacy.orders IS 'Legacy orders table with old naming conventions';
COMMENT ON COLUMN legacy.orders.cust_id IS 'Foreign key to customers table';
COMMENT ON COLUMN legacy.orders.order_dt IS 'Order date (maps to order_date in modern schema)';
COMMENT ON COLUMN legacy.orders.total_amt IS 'Total order amount (maps to total_amount in modern schema)';
COMMENT ON COLUMN legacy.orders.ship_addr IS 'Shipping address (maps to shipping_address in modern schema)';

-- =============================================================================
-- PRODUCTS TABLE (Legacy naming conventions)
-- =============================================================================
CREATE TABLE legacy.products (
    prod_id SERIAL PRIMARY KEY,
    prod_name VARCHAR(200) NOT NULL,
    prod_desc TEXT,
    unit_price DECIMAL(8,2) NOT NULL CHECK (unit_price >= 0),
    category_cd VARCHAR(10),
    active_flg BOOLEAN DEFAULT TRUE,
    created_dt DATE DEFAULT CURRENT_DATE,
    updated_dt DATE DEFAULT CURRENT_DATE
);

-- Add comments
COMMENT ON TABLE legacy.products IS 'Legacy products table with old naming conventions';
COMMENT ON COLUMN legacy.products.prod_id IS 'Legacy product ID (maps to product_id in modern schema)';
COMMENT ON COLUMN legacy.products.prod_name IS 'Product name (maps to name in modern schema)';
COMMENT ON COLUMN legacy.products.prod_desc IS 'Product description (maps to description in modern schema)';
COMMENT ON COLUMN legacy.products.unit_price IS 'Unit price (maps to price in modern schema)';
COMMENT ON COLUMN legacy.products.category_cd IS 'Category code (maps to category in modern schema with lookup)';
COMMENT ON COLUMN legacy.products.active_flg IS 'Active flag (maps to is_active in modern schema)';

-- =============================================================================
-- ORDER_ITEMS TABLE (Legacy naming conventions)
-- =============================================================================
CREATE TABLE legacy.order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES legacy.orders(order_id) ON DELETE CASCADE,
    prod_id INTEGER NOT NULL REFERENCES legacy.products(prod_id),
    qty INTEGER NOT NULL CHECK (qty > 0),
    unit_price DECIMAL(8,2) NOT NULL CHECK (unit_price >= 0),
    line_total DECIMAL(10,2) GENERATED ALWAYS AS (qty * unit_price) STORED
);

-- Add comments
COMMENT ON TABLE legacy.order_items IS 'Legacy order items (line items for orders)';
COMMENT ON COLUMN legacy.order_items.qty IS 'Quantity ordered';
COMMENT ON COLUMN legacy.order_items.line_total IS 'Calculated line total (qty * unit_price)';

-- =============================================================================
-- SUPPLIERS TABLE (Additional legacy table)
-- =============================================================================
CREATE TABLE legacy.suppliers (
    supplier_id SERIAL PRIMARY KEY,
    supplier_name VARCHAR(150) NOT NULL,
    contact_person VARCHAR(100),
    email_addr VARCHAR(255),
    phone_num VARCHAR(20),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_cd CHAR(2),
    zip_code VARCHAR(10),
    country_cd CHAR(3) DEFAULT 'USA',
    status_cd CHAR(1) DEFAULT 'A' CHECK (status_cd IN ('A', 'I')),
    created_dt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add comments
COMMENT ON TABLE legacy.suppliers IS 'Legacy suppliers table';
COMMENT ON COLUMN legacy.suppliers.state_cd IS 'Two-letter state code';
COMMENT ON COLUMN legacy.suppliers.country_cd IS 'Three-letter country code';

-- =============================================================================
-- INSERT SAMPLE DATA
-- =============================================================================

-- Insert sample customers
INSERT INTO legacy.customers (cust_name, email_addr, phone_num, credit_limit, last_login_dt, status_cd) VALUES
('John Smith', 'john.smith@email.com', '555-0101', 5000.00, '2024-01-15 10:30:00', 'A'),
('Jane Doe', 'jane.doe@email.com', '555-0102', 7500.00, '2024-01-16 14:22:00', 'A'),
('Bob Johnson', 'bob.johnson@email.com', '555-0103', 3000.00, '2024-01-14 09:15:00', 'A'),
('Alice Brown', 'alice.brown@email.com', '555-0104', 10000.00, '2024-01-17 16:45:00', 'A'),
('Charlie Wilson', 'charlie.wilson@email.com', '555-0105', 2500.00, '2024-01-13 11:20:00', 'S'),
('Diana Prince', 'diana.prince@email.com', '555-0106', 8000.00, '2024-01-18 08:30:00', 'A'),
('Frank Miller', 'frank.miller@email.com', '555-0107', 4500.00, '2024-01-12 15:45:00', 'I'),
('Grace Kelly', 'grace.kelly@email.com', '555-0108', 6000.00, '2024-01-19 12:15:00', 'A'),
('Henry Ford', 'henry.ford@email.com', '555-0109', 9000.00, '2024-01-11 17:20:00', 'A'),
('Ivy League', 'ivy.league@email.com', '555-0110', 3500.00, '2024-01-20 09:45:00', 'A');

-- Insert sample products
INSERT INTO legacy.products (prod_name, prod_desc, unit_price, category_cd, active_flg) VALUES
('Laptop Pro', 'High-performance laptop with 16GB RAM and 512GB SSD', 1299.99, 'TECH', TRUE),
('Wireless Mouse', 'Ergonomic wireless mouse with precision tracking', 29.99, 'TECH', TRUE),
('Office Chair', 'Comfortable ergonomic office chair with lumbar support', 199.99, 'FURN', TRUE),
('Desk Lamp', 'LED desk lamp with adjustable brightness and USB charging', 49.99, 'FURN', TRUE),
('Coffee Mug', 'Ceramic coffee mug with company logo', 12.99, 'MISC', TRUE),
('Monitor Stand', 'Adjustable monitor stand with storage compartment', 39.99, 'TECH', TRUE),
('Keyboard Mechanical', 'Mechanical keyboard with RGB backlighting', 89.99, 'TECH', TRUE),
('Notebook Set', 'Set of 3 professional notebooks', 24.99, 'MISC', TRUE),
('Desk Organizer', 'Bamboo desk organizer with multiple compartments', 34.99, 'FURN', TRUE),
('Webcam HD', 'HD webcam with auto-focus and noise cancellation', 79.99, 'TECH', FALSE);

-- Insert sample orders
INSERT INTO legacy.orders (cust_id, order_dt, total_amt, order_status, ship_addr) VALUES
(1, '2024-01-15', 1329.98, 'DELIVERED', '123 Main St, Springfield, IL 62701'),
(2, '2024-01-16', 249.98, 'SHIPPED', '456 Oak Ave, Chicago, IL 60601'),
(3, '2024-01-14', 62.98, 'DELIVERED', '789 Pine Rd, Peoria, IL 61602'),
(4, '2024-01-17', 1299.99, 'PROCESSING', '321 Elm St, Rockford, IL 61103'),
(5, '2024-01-13', 199.99, 'CANCELLED', '654 Maple Dr, Naperville, IL 60540'),
(6, '2024-01-18', 169.97, 'SHIPPED', '987 Cedar Ln, Aurora, IL 60506'),
(7, '2024-01-12', 89.99, 'DELIVERED', '147 Birch St, Joliet, IL 60435'),
(8, '2024-01-19', 234.97, 'PROCESSING', '258 Walnut Ave, Elgin, IL 60120'),
(9, '2024-01-11', 1419.97, 'DELIVERED', '369 Cherry St, Waukegan, IL 60085'),
(10, '2024-01-20', 79.99, 'PENDING', '741 Spruce Dr, Evanston, IL 60201');

-- Insert sample order items
INSERT INTO legacy.order_items (order_id, prod_id, qty, unit_price) VALUES
-- Order 1: Laptop + Mouse
(1, 1, 1, 1299.99),
(1, 2, 1, 29.99),
-- Order 2: Chair + Lamp
(2, 3, 1, 199.99),
(2, 4, 1, 49.99),
-- Order 3: Coffee Mugs
(3, 5, 5, 12.99),
-- Order 4: Laptop
(4, 1, 1, 1299.99),
-- Order 5: Chair (cancelled)
(5, 3, 1, 199.99),
-- Order 6: Monitor Stand + Organizer
(6, 6, 2, 39.99),
(6, 9, 3, 34.99),
-- Order 7: Keyboard
(7, 7, 1, 89.99),
-- Order 8: Notebook Sets
(8, 8, 3, 24.99),
(8, 4, 3, 49.99),
-- Order 9: Laptop + Keyboard + Mouse
(9, 1, 1, 1299.99),
(9, 7, 1, 89.99),
(9, 2, 1, 29.99),
-- Order 10: Webcam
(10, 10, 1, 79.99);

-- Insert sample suppliers
INSERT INTO legacy.suppliers (supplier_name, contact_person, email_addr, phone_num, address_line1, city, state_cd, zip_code, status_cd) VALUES
('TechCorp Industries', 'Michael Chen', 'mchen@techcorp.com', '800-555-0001', '1000 Technology Blvd', 'San Jose', 'CA', '95110', 'A'),
('Office Furniture Plus', 'Sarah Johnson', 'sjohnson@officefurn.com', '800-555-0002', '2500 Furniture Ave', 'Grand Rapids', 'MI', '49503', 'A'),
('Global Electronics', 'David Kim', 'dkim@globalelec.com', '800-555-0003', '750 Electronics Way', 'Austin', 'TX', '78701', 'A'),
('Workspace Solutions', 'Lisa Wang', 'lwang@workspace.com', '800-555-0004', '1200 Business Park Dr', 'Atlanta', 'GA', '30309', 'I'),
('Premium Supplies Co', 'Robert Martinez', 'rmartinez@premiumsup.com', '800-555-0005', '3000 Supply Chain St', 'Denver', 'CO', '80202', 'A');

-- =============================================================================
-- CREATE INDEXES FOR BETTER PERFORMANCE
-- =============================================================================

-- Customer indexes
CREATE INDEX idx_customers_email ON legacy.customers(email_addr);
CREATE INDEX idx_customers_status ON legacy.customers(status_cd);
CREATE INDEX idx_customers_created ON legacy.customers(created_dt);

-- Order indexes
CREATE INDEX idx_orders_customer ON legacy.orders(cust_id);
CREATE INDEX idx_orders_date ON legacy.orders(order_dt);
CREATE INDEX idx_orders_status ON legacy.orders(order_status);
CREATE INDEX idx_orders_created ON legacy.orders(created_ts);

-- Product indexes
CREATE INDEX idx_products_category ON legacy.products(category_cd);
CREATE INDEX idx_products_active ON legacy.products(active_flg);
CREATE INDEX idx_products_name ON legacy.products(prod_name);

-- Order items indexes
CREATE INDEX idx_order_items_order ON legacy.order_items(order_id);
CREATE INDEX idx_order_items_product ON legacy.order_items(prod_id);

-- Supplier indexes
CREATE INDEX idx_suppliers_status ON legacy.suppliers(status_cd);
CREATE INDEX idx_suppliers_name ON legacy.suppliers(supplier_name);

-- =============================================================================
-- CREATE VIEWS FOR EASIER QUERYING
-- =============================================================================

-- Customer summary view
CREATE VIEW legacy.customer_summary AS
SELECT 
    c.cust_id,
    c.cust_name,
    c.email_addr,
    c.status_cd,
    c.credit_limit,
    COUNT(o.order_id) as total_orders,
    COALESCE(SUM(o.total_amt), 0) as total_spent,
    MAX(o.order_dt) as last_order_date
FROM legacy.customers c
LEFT JOIN legacy.orders o ON c.cust_id = o.cust_id
GROUP BY c.cust_id, c.cust_name, c.email_addr, c.status_cd, c.credit_limit;

-- Product sales view
CREATE VIEW legacy.product_sales AS
SELECT 
    p.prod_id,
    p.prod_name,
    p.category_cd,
    p.unit_price,
    COALESCE(SUM(oi.qty), 0) as total_sold,
    COALESCE(SUM(oi.line_total), 0) as total_revenue,
    COUNT(DISTINCT oi.order_id) as orders_count
FROM legacy.products p
LEFT JOIN legacy.order_items oi ON p.prod_id = oi.prod_id
GROUP BY p.prod_id, p.prod_name, p.category_cd, p.unit_price;

-- Order details view
CREATE VIEW legacy.order_details AS
SELECT 
    o.order_id,
    o.cust_id,
    c.cust_name,
    o.order_dt,
    o.order_status,
    o.total_amt,
    COUNT(oi.item_id) as item_count,
    SUM(oi.qty) as total_quantity
FROM legacy.orders o
JOIN legacy.customers c ON o.cust_id = c.cust_id
LEFT JOIN legacy.order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.cust_id, c.cust_name, o.order_dt, o.order_status, o.total_amt;

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant all privileges on schema to the source user
GRANT ALL PRIVILEGES ON SCHEMA legacy TO source_user;

-- Grant all privileges on all tables in schema
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA legacy TO source_user;

-- Grant all privileges on all sequences in schema
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA legacy TO source_user;

-- Grant usage on all views
GRANT SELECT ON ALL TABLES IN SCHEMA legacy TO source_user;

-- =============================================================================
-- STATISTICS AND ANALYSIS
-- =============================================================================

-- Update table statistics for query optimization
ANALYZE legacy.customers;
ANALYZE legacy.orders;
ANALYZE legacy.products;
ANALYZE legacy.order_items;
ANALYZE legacy.suppliers;

-- =============================================================================
-- DATA QUALITY CHECKS (Optional - for validation)
-- =============================================================================

-- Check for data quality issues that might affect migration
DO $$
BEGIN
    -- Check for customers with invalid email formats
    IF EXISTS (SELECT 1 FROM legacy.customers WHERE email_addr IS NOT NULL AND email_addr !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') THEN
        RAISE WARNING 'Found customers with invalid email formats';
    END IF;
    
    -- Check for orders without items
    IF EXISTS (SELECT 1 FROM legacy.orders o WHERE NOT EXISTS (SELECT 1 FROM legacy.order_items oi WHERE oi.order_id = o.order_id)) THEN
        RAISE WARNING 'Found orders without any items';
    END IF;
    
    -- Check for products with zero or negative prices
    IF EXISTS (SELECT 1 FROM legacy.products WHERE unit_price <= 0) THEN
        RAISE WARNING 'Found products with zero or negative prices';
    END IF;
    
    RAISE INFO 'Legacy database initialization completed successfully';
    RAISE INFO 'Total customers: %', (SELECT COUNT(*) FROM legacy.customers);
    RAISE INFO 'Total orders: %', (SELECT COUNT(*) FROM legacy.orders);
    RAISE INFO 'Total products: %', (SELECT COUNT(*) FROM legacy.products);
    RAISE INFO 'Total suppliers: %', (SELECT COUNT(*) FROM legacy.suppliers);
END $$;
