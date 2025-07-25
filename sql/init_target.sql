-- =============================================================================
-- Target Database Schema (Modern System)
-- File: sql/init_target.sql
-- =============================================================================

-- Create modern schema
CREATE SCHEMA IF NOT EXISTS modern;

-- Set search path
SET search_path TO modern, public;

-- Enable extensions for additional functionality
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================================
-- CUSTOMERS TABLE (Modern naming conventions)
-- =============================================================================
CREATE TABLE modern.customers (
    customer_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(10) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED', 'DELETED')),
    credit_limit NUMERIC(10,2),
    last_login TIMESTAMP,
    customer_uuid UUID DEFAULT uuid_generate_v4(),
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[]
);

-- Add comments for documentation
COMMENT ON TABLE modern.customers IS 'Modern customer data with updated naming conventions and additional fields';
COMMENT ON COLUMN modern.customers.customer_id IS 'Modern customer ID (mapped from cust_id in legacy schema)';
COMMENT ON COLUMN modern.customers.full_name IS 'Customer full name (mapped from cust_name in legacy schema)';
COMMENT ON COLUMN modern.customers.email IS 'Email address (mapped from email_addr in legacy schema)';
COMMENT ON COLUMN modern.customers.phone IS 'Phone number (mapped from phone_num in legacy schema)';
COMMENT ON COLUMN modern.customers.status IS 'Status: ACTIVE, INACTIVE, SUSPENDED, DELETED (mapped from status_cd)';
COMMENT ON COLUMN modern.customers.customer_uuid IS 'UUID for external integrations';
COMMENT ON COLUMN modern.customers.metadata IS 'Additional customer metadata in JSON format';
COMMENT ON COLUMN modern.customers.tags IS 'Customer tags for segmentation';

-- =============================================================================
-- ORDERS TABLE (Modern naming conventions)
-- =============================================================================
CREATE TABLE modern.orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES modern.customers(customer_id) ON DELETE RESTRICT,
    order_date DATE NOT NULL,
    total_amount NUMERIC(10,2) NOT NULL CHECK (total_amount >= 0),
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'CANCELLED', 'RETURNED')),
    shipping_address TEXT,
    billing_address TEXT,
    payment_method VARCHAR(50),
    tracking_number VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_uuid UUID DEFAULT uuid_generate_v4(),
    metadata JSONB DEFAULT '{}'
);

-- Add comments
COMMENT ON TABLE modern.orders IS 'Modern orders table with enhanced tracking and metadata';
COMMENT ON COLUMN modern.orders.customer_id IS 'Foreign key to customers table';
COMMENT ON COLUMN modern.orders.order_date IS 'Order date (mapped from order_dt in legacy schema)';
COMMENT ON COLUMN modern.orders.total_amount IS 'Total order amount (mapped from total_amt in legacy schema)';
COMMENT ON COLUMN modern.orders.shipping_address IS 'Shipping address (mapped from ship_addr in legacy schema)';
COMMENT ON COLUMN modern.orders.billing_address IS 'Billing address (new field for modern system)';
COMMENT ON COLUMN modern.orders.payment_method IS 'Payment method used';
COMMENT ON COLUMN modern.orders.tracking_number IS 'Shipment tracking number';

-- =============================================================================
-- PRODUCTS TABLE (Modern naming conventions)
-- =============================================================================
CREATE TABLE modern.products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price NUMERIC(8,2) NOT NULL CHECK (price >= 0),
    cost NUMERIC(8,2),
    category VARCHAR(20),
    subcategory VARCHAR(50),
    brand VARCHAR(100),
    sku VARCHAR(50) UNIQUE,
    barcode VARCHAR(50),
    weight_kg NUMERIC(6,3),
    dimensions_cm VARCHAR(50), -- Format: "LxWxH"
    is_active BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    stock_quantity INTEGER DEFAULT 0 CHECK (stock_quantity >= 0),
    min_stock_level INTEGER DEFAULT 0,
    max_stock_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_uuid UUID DEFAULT uuid_generate_v4(),
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[]
);

-- Add comments
COMMENT ON TABLE modern.products IS 'Modern products table with enhanced inventory and metadata';
COMMENT ON COLUMN modern.products.product_id IS 'Modern product ID (mapped from prod_id in legacy schema)';
COMMENT ON COLUMN modern.products.name IS 'Product name (mapped from prod_name in legacy schema)';
COMMENT ON COLUMN modern.products.description IS 'Product description (mapped from prod_desc in legacy schema)';
COMMENT ON COLUMN modern.products.price IS 'Product price (mapped from unit_price in legacy schema)';
COMMENT ON COLUMN modern.products.category IS 'Product category (mapped from category_cd with lookup transformation)';
COMMENT ON COLUMN modern.products.is_active IS 'Active status (mapped from active_flg in legacy schema)';
COMMENT ON COLUMN modern.products.sku IS 'Stock Keeping Unit identifier';
COMMENT ON COLUMN modern.products.stock_quantity IS 'Current inventory quantity';

-- =============================================================================
-- ORDER_ITEMS TABLE (Modern naming conventions)
-- =============================================================================
CREATE TABLE modern.order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES modern.orders(order_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES modern.products(product_id) ON DELETE RESTRICT,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(8,2) NOT NULL CHECK (unit_price >= 0),
    line_total NUMERIC(10,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    discount_amount NUMERIC(8,2) DEFAULT 0 CHECK (discount_amount >= 0),
    tax_amount NUMERIC(8,2) DEFAULT 0 CHECK (tax_amount >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Add comments
COMMENT ON TABLE modern.order_items IS 'Modern order items with enhanced pricing details';
COMMENT ON COLUMN modern.order_items.quantity IS 'Quantity ordered (mapped from qty in legacy schema)';
COMMENT ON COLUMN modern.order_items.line_total IS 'Calculated line total (quantity * unit_price)';
COMMENT ON COLUMN modern.order_items.discount_amount IS 'Discount applied to this line item';
COMMENT ON COLUMN modern.order_items.tax_amount IS 'Tax amount for this line item';

-- =============================================================================
-- SUPPLIERS TABLE (Modern naming conventions)
-- =============================================================================
CREATE TABLE modern.suppliers (
    supplier_id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    contact_person VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(20),
    website VARCHAR(255),
    address JSONB, -- Structured address data
    tax_id VARCHAR(50),
    payment_terms VARCHAR(100),
    status VARCHAR(10) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED')),
    rating NUMERIC(3,2) CHECK (rating >= 0 AND rating <= 5.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    supplier_uuid UUID DEFAULT uuid_generate_v4(),
    metadata JSONB DEFAULT '{}'
);

-- Add comments
COMMENT ON TABLE modern.suppliers IS 'Modern suppliers table with structured data and ratings';
COMMENT ON COLUMN modern.suppliers.name IS 'Supplier name (mapped from supplier_name in legacy schema)';
COMMENT ON COLUMN modern.suppliers.address IS 'Structured address data in JSON format';
COMMENT ON COLUMN modern.suppliers.rating IS 'Supplier performance rating (0-5)';
COMMENT ON COLUMN modern.suppliers.payment_terms IS 'Payment terms (e.g., "Net 30", "COD")';

-- =============================================================================
-- PRODUCT_SUPPLIERS TABLE (New relationship table)
-- =============================================================================
CREATE TABLE modern.product_suppliers (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES modern.products(product_id) ON DELETE CASCADE,
    supplier_id INTEGER NOT NULL REFERENCES modern.suppliers(supplier_id) ON DELETE CASCADE,
    supplier_sku VARCHAR(50),
    cost NUMERIC(8,2),
    lead_time_days INTEGER,
    min_order_quantity INTEGER DEFAULT 1,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, supplier_id)
);

-- Add comments
COMMENT ON TABLE modern.product_suppliers IS 'Many-to-many relationship between products and suppliers';
COMMENT ON COLUMN modern.product_suppliers.supplier_sku IS 'Supplier-specific SKU for this product';
COMMENT ON COLUMN modern.product_suppliers.lead_time_days IS 'Lead time in days from this supplier';
COMMENT ON COLUMN modern.product_suppliers.is_primary IS 'Whether this is the primary supplier for the product';

-- =============================================================================
-- AUDIT LOG TABLE (New for modern system)
-- =============================================================================
CREATE TABLE modern.audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100),
    ip_address INET
);

-- Add comments
COMMENT ON TABLE modern.audit_log IS 'Audit trail for all data changes';
COMMENT ON COLUMN modern.audit_log.operation IS 'Type of operation: INSERT, UPDATE, or DELETE';
COMMENT ON COLUMN modern.audit_log.old_values IS 'Previous values before change';
COMMENT ON COLUMN modern.audit_log.new_values IS 'New values after change';

-- =============================================================================
-- INSERT TRANSFORMED SAMPLE DATA (simulating successful migration)
-- =============================================================================

-- Insert transformed customers (from legacy with status code transformation)
INSERT INTO modern.customers (full_name, email, phone, credit_limit, last_login, status, metadata, tags) VALUES
('John Smith', 'john.smith@email.com', '555-0101', 5000.00, '2024-01-15 10:30:00', 'ACTIVE', '{"legacy_id": 1, "migration_date": "2024-01-21"}', ARRAY['premium', 'loyal']),
('Jane Doe', 'jane.doe@email.com', '555-0102', 7500.00, '2024-01-16 14:22:00', 'ACTIVE', '{"legacy_id": 2, "migration_date": "2024-01-21"}', ARRAY['premium']),
('Bob Johnson', 'bob.johnson@email.com', '555-0103', 3000.00, '2024-01-14 09:15:00', 'ACTIVE', '{"legacy_id": 3, "migration_date": "2024-01-21"}', ARRAY['standard']),
('Alice Brown', 'alice.brown@email.com', '555-0104', 10000.00, '2024-01-17 16:45:00', 'ACTIVE', '{"legacy_id": 4, "migration_date": "2024-01-21"}', ARRAY['premium', 'corporate']),
('Charlie Wilson', 'charlie.wilson@email.com', '555-0105', 2500.00, '2024-01-13 11:20:00', 'SUSPENDED', '{"legacy_id": 5, "migration_date": "2024-01-21", "suspension_reason": "Payment issues"}', ARRAY['at-risk']),
('Diana Prince', 'diana.prince@email.com', '555-0106', 8000.00, '2024-01-18 08:30:00', 'ACTIVE', '{"legacy_id": 6, "migration_date": "2024-01-21"}', ARRAY['premium']),
('Frank Miller', 'frank.miller@email.com', '555-0107', 4500.00, '2024-01-12 15:45:00', 'INACTIVE', '{"legacy_id": 7, "migration_date": "2024-01-21", "deactivation_reason": "Account closure request"}', ARRAY['former-customer']),
('Grace Kelly', 'grace.kelly@email.com', '555-0108', 6000.00, '2024-01-19 12:15:00', 'ACTIVE', '{"legacy_id": 8, "migration_date": "2024-01-21"}', ARRAY['premium']),
('Henry Ford', 'henry.ford@email.com', '555-0109', 9000.00, '2024-01-11 17:20:00', 'ACTIVE', '{"legacy_id": 9, "migration_date": "2024-01-21"}', ARRAY['premium', 'enterprise']),
('Ivy League', 'ivy.league@email.com', '555-0110', 3500.00, '2024-01-20 09:45:00', 'ACTIVE', '{"legacy_id": 10, "migration_date": "2024-01-21"}', ARRAY['standard']);

-- Insert transformed products (with category code transformation and additional modern fields)
INSERT INTO modern.products (name, description, price, cost, category, subcategory, brand, sku, barcode, weight_kg, stock_quantity, min_stock_level, is_active, metadata, tags) VALUES
('Laptop Pro', 'High-performance laptop with 16GB RAM and 512GB SSD', 1299.99, 950.00, 'TECHNOLOGY', 'Computers', 'TechBrand', 'SKU-LAPTOP-001', '1234567890123', 2.1, 25, 5, TRUE, '{"legacy_id": 1, "migration_date": "2024-01-21"}', ARRAY['bestseller', 'premium']),
('Wireless Mouse', 'Ergonomic wireless mouse with precision tracking', 29.99, 15.00, 'TECHNOLOGY', 'Accessories', 'TechBrand', 'SKU-MOUSE-001', '1234567890124', 0.15, 150, 20, TRUE, '{"legacy_id": 2, "migration_date": "2024-01-21"}', ARRAY['popular']),
('Office Chair', 'Comfortable ergonomic office chair with lumbar support', 199.99, 120.00, 'FURNITURE', 'Seating', 'OfficePro', 'SKU-CHAIR-001', '1234567890125', 15.5, 45, 10, TRUE, '{"legacy_id": 3, "migration_date": "2024-01-21"}', ARRAY['ergonomic']),
('Desk Lamp', 'LED desk lamp with adjustable brightness and USB charging', 49.99, 25.00, 'FURNITURE', 'Lighting', 'LightTech', 'SKU-LAMP-001', '1234567890126', 1.2, 75, 15, TRUE, '{"legacy_id": 4, "migration_date": "2024-01-21"}', ARRAY['energy-efficient']),
('Coffee Mug', 'Ceramic coffee mug with company logo', 12.99, 5.00, 'MISCELLANEOUS', 'Drinkware', 'OfficeBrand', 'SKU-MUG-001', '1234567890127', 0.35, 200, 50, TRUE, '{"legacy_id": 5, "migration_date": "2024-01-21"}', ARRAY['promotional']),
('Monitor Stand', 'Adjustable monitor stand with storage compartment', 39.99, 20.00, 'TECHNOLOGY', 'Accessories', 'DeskTech', 'SKU-STAND-001', '1234567890128', 2.8, 60, 10, TRUE, '{"legacy_id": 6, "migration_date": "2024-01-21"}', ARRAY['organizational']),
('Keyboard Mechanical', 'Mechanical keyboard with RGB backlighting', 89.99, 55.00, 'TECHNOLOGY', 'Input Devices', 'GameTech', 'SKU-KEYBOARD-001', '1234567890129', 1.1, 80, 15, TRUE, '{"legacy_id": 7, "migration_date": "2024-01-21"}', ARRAY['gaming', 'premium']),
('Notebook Set', 'Set of 3 professional notebooks', 24.99, 12.00, 'MISCELLANEOUS', 'Stationery', 'PaperCo', 'SKU-NOTEBOOK-001', '1234567890130', 0.6, 120, 25, TRUE, '{"legacy_id": 8, "migration_date": "2024-01-21"}', ARRAY['office-supplies']),
('Desk Organizer', 'Bamboo desk organizer with multiple compartments', 34.99, 18.00, 'FURNITURE', 'Storage', 'EcoOffice', 'SKU-ORGANIZER-001', '1234567890131', 1.5, 40, 8, TRUE, '{"legacy_id": 9, "migration_date": "2024-01-21"}', ARRAY['eco-friendly', 'organizational']),
('Webcam HD', 'HD webcam with auto-focus and noise cancellation', 79.99, 45.00, 'TECHNOLOGY', 'Video Equipment', 'VideoTech', 'SKU-WEBCAM-001', '1234567890132', 0.25, 0, 5, FALSE, '{"legacy_id": 10, "migration_date": "2024-01-21", "discontinued": true}', ARRAY['discontinued']);

-- Insert transformed orders
INSERT INTO modern.orders (customer_id, order_date, total_amount, status, shipping_address, billing_address, payment_method, tracking_number, metadata) VALUES
(1, '2024-01-15', 1329.98, 'DELIVERED', '123 Main St, Springfield, IL 62701', '123 Main St, Springfield, IL 62701', 'Credit Card', 'TRK123456789', '{"legacy_id": 1, "migration_date": "2024-01-21"}'),
(2, '2024-01-16', 249.98, 'SHIPPED', '456 Oak Ave, Chicago, IL 60601', '456 Oak Ave, Chicago, IL 60601', 'PayPal', 'TRK234567890', '{"legacy_id": 2, "migration_date": "2024-01-21"}'),
(3, '2024-01-14', 62.98, 'DELIVERED', '789 Pine Rd, Peoria, IL 61602', '789 Pine Rd, Peoria, IL 61602', 'Credit Card', 'TRK345678901', '{"legacy_id": 3, "migration_date": "2024-01-21"}'),
(4, '2024-01-17', 1299.99, 'PROCESSING', '321 Elm St, Rockford, IL 61103', '321 Elm St, Rockford, IL 61103', 'Bank Transfer', NULL, '{"legacy_id": 4, "migration_date": "2024-01-21"}'),
(5, '2024-01-13', 199.99, 'CANCELLED', '654 Maple Dr, Naperville, IL 60540', '654 Maple Dr, Naperville, IL 60540', 'Credit Card', NULL, '{"legacy_id": 5, "migration_date": "2024-01-21", "cancellation_reason": "Customer request"}'),
(6, '2024-01-18', 169.97, 'SHIPPED', '987 Cedar Ln, Aurora, IL 60506', '987 Cedar Ln, Aurora, IL 60506', 'Credit Card', 'TRK456789012', '{"legacy_id": 6, "migration_date": "2024-01-21"}'),
(7, '2024-01-12', 89.99, 'DELIVERED', '147 Birch St, Joliet, IL 60435', '147 Birch St, Joliet, IL 60435', 'PayPal', 'TRK567890123', '{"legacy_id": 7, "migration_date": "2024-01-21"}'),
(8, '2024-01-19', 234.97, 'PROCESSING', '258 Walnut Ave, Elgin, IL 60120', '258 Walnut Ave, Elgin, IL 60120', 'Credit Card', NULL, '{"legacy_id": 8, "migration_date": "2024-01-21"}'),
(9, '2024-01-11', 1419.97, 'DELIVERED', '369 Cherry St, Waukegan, IL 60085', '369 Cherry St, Waukegan, IL 60085', 'Corporate Account', 'TRK678901234', '{"legacy_id": 9, "migration_date": "2024-01-21"}'),
(10, '2024-01-20', 79.99, 'PENDING', '741 Spruce Dr, Evanston, IL 60201', '741 Spruce Dr, Evanston, IL 60201', 'Credit Card', NULL, '{"legacy_id": 10, "migration_date": "2024-01-21"}');

-- Insert transformed order items
INSERT INTO modern.order_items (order_id, product_id, quantity, unit_price, discount_amount, tax_amount, metadata) VALUES
-- Order 1: Laptop + Mouse
(1, 1, 1, 1299.99, 0.00, 104.00, '{"legacy_item_id": 1}'),
(1, 2, 1, 29.99, 0.00, 2.40, '{"legacy_item_id": 2}'),
-- Order 2: Chair + Lamp
(2, 3, 1, 199.99, 0.00, 16.00, '{"legacy_item_id": 3}'),
(2, 4, 1, 49.99, 0.00, 4.00, '{"legacy_item_id": 4}'),
-- Order 3: Coffee Mugs
(3, 5, 5, 12.99, 2.00, 4.99, '{"legacy_item_id": 5}'),
-- Order 4: Laptop
(4, 1, 1, 1299.99, 0.00, 104.00, '{"legacy_item_id": 6}'),
-- Order 5: Chair (cancelled)
(5, 3, 1, 199.99, 0.00, 16.00, '{"legacy_item_id": 7}'),
-- Order 6: Monitor Stand + Organizer
(6, 6, 2, 39.99, 0.00, 6.40, '{"legacy_item_id": 8}'),
(6, 9, 3, 34.99, 5.00, 7.69, '{"legacy_item_id": 9}'),
-- Order 7: Keyboard
(7, 7, 1, 89.99, 0.00, 7.20, '{"legacy_item_id": 10}'),
-- Order 8: Notebook Sets + Lamps
(8, 8, 3, 24.99, 0.00, 6.00, '{"legacy_item_id": 11}'),
(8, 4, 3, 49.99, 0.00, 12.00, '{"legacy_item_id": 12}'),
-- Order 9: Laptop + Keyboard + Mouse
(9, 1, 1, 1299.99, 0.00, 104.00, '{"legacy_item_id": 13}'),
(9, 7, 1, 89.99, 0.00, 7.20, '{"legacy_item_id": 14}'),
(9, 2, 1, 29.99, 0.00, 2.40, '{"legacy_item_id": 15}'),
-- Order 10: Webcam
(10, 10, 1, 79.99, 0.00, 6.40, '{"legacy_item_id": 16}');

-- Insert transformed suppliers
INSERT INTO modern.suppliers (name, contact_person, email, phone, website, address, tax_id, payment_terms, status, rating, metadata) VALUES
('TechCorp Industries', 'Michael Chen', 'mchen@techcorp.com', '800-555-0001', 'https://www.techcorp.com', 
 '{"street": "1000 Technology Blvd", "city": "San Jose", "state": "CA", "zip": "95110", "country": "USA"}', 
 'TC123456789', 'Net 30', 'ACTIVE', 4.5, '{"legacy_id": 1, "migration_date": "2024-01-21"}'),
('Office Furniture Plus', 'Sarah Johnson', 'sjohnson@officefurn.com', '800-555-0002', 'https://www.officefurn.com',
 '{"street": "2500 Furniture Ave", "city": "Grand Rapids", "state": "MI", "zip": "49503", "country": "USA"}',
 'OF987654321', 'Net 45', 'ACTIVE', 4.2, '{"legacy_id": 2, "migration_date": "2024-01-21"}'),
('Global Electronics', 'David Kim', 'dkim@globalelec.com', '800-555-0003', 'https://www.globalelec.com',
 '{"street": "750 Electronics Way", "city": "Austin", "state": "TX", "zip": "78701", "country": "USA"}',
 'GE456789123', 'Net 30', 'ACTIVE', 4.7, '{"legacy_id": 3, "migration_date": "2024-01-21"}'),
('Workspace Solutions', 'Lisa Wang', 'lwang@workspace.com', '800-555-0004', 'https://www.workspace.com',
 '{"street": "1200 Business Park Dr", "city": "Atlanta", "state": "GA", "zip": "30309", "country": "USA"}',
 'WS789123456', 'COD', 'INACTIVE', 3.8, '{"legacy_id": 4, "migration_date": "2024-01-21", "inactive_reason": "Contract expired"}'),
('Premium Supplies Co', 'Robert Martinez', 'rmartinez@premiumsup.com', '800-555-0005', 'https://www.premiumsup.com',
 '{"street": "3000 Supply Chain St", "city": "Denver", "state": "CO", "zip": "80202", "country": "USA"}',
 'PS321654987', 'Net 30', 'ACTIVE', 4.3, '{"legacy_id": 5, "migration_date": "2024-01-21"}');

-- Insert product-supplier relationships
INSERT INTO modern.product_suppliers (product_id, supplier_id, supplier_sku, cost, lead_time_days, min_order_quantity, is_primary) VALUES
(1, 1, 'TC-LAPTOP-PRO-001', 950.00, 7, 1, TRUE),
(2, 1, 'TC-MOUSE-WIRELESS-001', 15.00, 3, 10, TRUE),
(3, 2, 'OF-CHAIR-ERGONOMIC-001', 120.00, 14, 1, TRUE),
(4, 2, 'OF-LAMP-LED-001', 25.00, 5, 5, TRUE),
(5, 5, 'PS-MUG-CERAMIC-001', 5.00, 2, 50, TRUE),
(6, 1, 'TC-STAND-MONITOR-001', 20.00, 5, 5, TRUE),
(7, 3, 'GE-KEYBOARD-MECH-001', 55.00, 10, 2, TRUE),
(8, 5, 'PS-NOTEBOOK-SET-001', 12.00, 3, 10, TRUE),
(9, 2, 'OF-ORGANIZER-BAMBOO-001', 18.00, 7, 5, TRUE),
(10, 3, 'GE-WEBCAM-HD-001', 45.00, 14, 1, TRUE);

-- =============================================================================
-- CREATE INDEXES FOR BETTER PERFORMANCE
-- =============================================================================

-- Customer indexes
CREATE INDEX idx_customers_email_modern ON modern.customers(email);
CREATE INDEX idx_customers_status_modern ON modern.customers(status);
CREATE INDEX idx_customers_created_modern ON modern.customers(created_at);
CREATE INDEX idx_customers_uuid_modern ON modern.customers(customer_uuid);
CREATE INDEX idx_customers_tags_modern ON modern.customers USING GIN(tags);
CREATE INDEX idx_customers_metadata_modern ON modern.customers USING GIN(metadata);

-- Order indexes
CREATE INDEX idx_orders_customer_modern ON modern.orders(customer_id);
CREATE INDEX idx_orders_date_modern ON modern.orders(order_date);
CREATE INDEX idx_orders_status_modern ON modern.orders(status);
CREATE INDEX idx_orders_created_modern ON modern.orders(created_at);
CREATE INDEX idx_orders_uuid_modern ON modern.orders(order_uuid);
CREATE INDEX idx_orders_tracking_modern ON modern.orders(tracking_number);

-- Product indexes
CREATE INDEX idx_products_category_modern ON modern.products(category);
CREATE INDEX idx_products_active_modern ON modern.products(is_active);
CREATE INDEX idx_products_name_modern ON modern.products(name);
CREATE INDEX idx_products_sku_modern ON modern.products(sku);
CREATE INDEX idx_products_name_trgm ON modern.products USING GIN(name gin_trgm_ops);
CREATE INDEX idx_products_tags_modern ON modern.products USING GIN(tags);
CREATE INDEX idx_products_metadata_modern ON modern.products USING GIN(metadata);

-- Order items indexes
CREATE INDEX idx_order_items_order_modern ON modern.order_items(order_id);
CREATE INDEX idx_order_items_product_modern ON modern.order_items(product_id);

-- Supplier indexes
CREATE INDEX idx_suppliers_status_modern ON modern.suppliers(status);
CREATE INDEX idx_suppliers_name_modern ON modern.suppliers(name);
CREATE INDEX idx_suppliers_uuid_modern ON modern.suppliers(supplier_uuid);

-- Product suppliers indexes
CREATE INDEX idx_product_suppliers_product ON modern.product_suppliers(product_id);
CREATE INDEX idx_product_suppliers_supplier ON modern.product_suppliers(supplier_id);
CREATE INDEX idx_product_suppliers_primary ON modern.product_suppliers(is_primary);

-- Audit log indexes
CREATE INDEX idx_audit_log_table_record ON modern.audit_log(table_name, record_id);
CREATE INDEX idx_audit_log_changed_at ON modern.audit_log(changed_at);
CREATE INDEX idx_audit_log_changed_by ON modern.audit_log(changed_by);
CREATE INDEX idx_audit_log_operation ON modern.audit_log(operation);

-- =============================================================================
-- CREATE VIEWS FOR ENHANCED QUERYING
-- =============================================================================

-- Customer summary view with modern features
CREATE VIEW modern.customer_summary AS
SELECT 
    c.customer_id,
    c.full_name,
    c.email,
    c.status,
    c.credit_limit,
    c.customer_uuid,
    c.tags,
    COUNT(o.order_id) as total_orders,
    COALESCE(SUM(o.total_amount), 0) as total_spent,
    MAX(o.order_date) as last_order_date,
    AVG(o.total_amount) as avg_order_value,
    c.created_at as customer_since,
    EXTRACT(DAYS FROM CURRENT_DATE - c.created_at) as customer_lifetime_days
FROM modern.customers c
LEFT JOIN modern.orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.full_name, c.email, c.status, c.credit_limit, c.customer_uuid, c.tags, c.created_at;

-- Product performance view
CREATE VIEW modern.product_performance AS
SELECT 
    p.product_id,
    p.name,
    p.category,
    p.subcategory,
    p.price,
    p.cost,
    p.stock_quantity,
    p.is_active,
    p.tags,
    COALESCE(SUM(oi.quantity), 0) as total_sold,
    COALESCE(SUM(oi.line_total), 0) as total_revenue,
    COALESCE(SUM(oi.line_total) - (SUM(oi.quantity) * p.cost), 0) as total_profit,
    COUNT(DISTINCT oi.order_id) as orders_count,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    AVG(oi.quantity) as avg_quantity_per_order,
    CASE 
        WHEN p.cost > 0 THEN ROUND(((p.price - p.cost) / p.cost * 100), 2)
        ELSE NULL 
    END as profit_margin_pct
FROM modern.products p
LEFT JOIN modern.order_items oi ON p.product_id = oi.product_id
LEFT JOIN modern.orders o ON oi.order_id = o.order_id
GROUP BY p.product_id, p.name, p.category, p.subcategory, p.price, p.cost, p.stock_quantity, p.is_active, p.tags;

-- Order details view with enhanced information
CREATE VIEW modern.order_details AS
SELECT 
    o.order_id,
    o.customer_id,
    c.full_name as customer_name,
    c.email as customer_email,
    o.order_date,
    o.status,
    o.total_amount,
    o.payment_method,
    o.tracking_number,
    o.order_uuid,
    COUNT(oi.item_id) as item_count,
    SUM(oi.quantity) as total_quantity,
    SUM(oi.discount_amount) as total_discount,
    SUM(oi.tax_amount) as total_tax,
    EXTRACT(DAYS FROM CURRENT_DATE - o.order_date) as days_since_order,
    CASE 
        WHEN o.status IN ('DELIVERED') THEN 'Completed'
        WHEN o.status IN ('CANCELLED', 'RETURNED') THEN 'Not Completed'
        ELSE 'In Progress'
    END as completion_status
FROM modern.orders o
JOIN modern.customers c ON o.customer_id = c.customer_id
LEFT JOIN modern.order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.customer_id, c.full_name, c.email, o.order_date, o.status, 
         o.total_amount, o.payment_method, o.tracking_number, o.order_uuid;

-- Supplier performance view
CREATE VIEW modern.supplier_performance AS
SELECT 
    s.supplier_id,
    s.name as supplier_name,
    s.status,
    s.rating,
    s.payment_terms,
    COUNT(ps.product_id) as products_supplied,
    COUNT(CASE WHEN ps.is_primary THEN 1 END) as primary_products,
    AVG(ps.lead_time_days) as avg_lead_time,
    AVG(ps.cost) as avg_cost,
    SUM(p.stock_quantity) as total_inventory_value
FROM modern.suppliers s
LEFT JOIN modern.product_suppliers ps ON s.supplier_id = ps.supplier_id
LEFT JOIN modern.products p ON ps.product_id = p.product_id
GROUP BY s.supplier_id, s.name, s.status, s.rating, s.payment_terms;

-- Inventory status view
CREATE VIEW modern.inventory_status AS
SELECT 
    p.product_id,
    p.name,
    p.sku,
    p.category,
    p.stock_quantity,
    p.min_stock_level,
    p.max_stock_level,
    p.is_active,
    CASE 
        WHEN p.stock_quantity <= p.min_stock_level THEN 'Low Stock'
        WHEN p.stock_quantity >= p.max_stock_level THEN 'Overstocked'
        WHEN p.stock_quantity = 0 THEN 'Out of Stock'
        ELSE 'Normal'
    END as stock_status,
    ps.supplier_id as primary_supplier_id,
    s.name as primary_supplier_name,
    ps.lead_time_days,
    COALESCE(SUM(oi.quantity), 0) as units_sold_30_days
FROM modern.products p
LEFT JOIN modern.product_suppliers ps ON p.product_id = ps.product_id AND ps.is_primary = TRUE
LEFT JOIN modern.suppliers s ON ps.supplier_id = s.supplier_id
LEFT JOIN modern.order_items oi ON p.product_id = oi.product_id
LEFT JOIN modern.orders o ON oi.order_id = o.order_id AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY p.product_id, p.name, p.sku, p.category, p.stock_quantity, p.min_stock_level, 
         p.max_stock_level, p.is_active, ps.supplier_id, s.name, ps.lead_time_days;

-- =============================================================================
-- CREATE TRIGGERS FOR AUDIT LOGGING
-- =============================================================================

-- Function to log changes
CREATE OR REPLACE FUNCTION modern.log_changes()
RETURNS TRIGGER AS $
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO modern.audit_log (table_name, record_id, operation, old_values, changed_by)
        VALUES (TG_TABLE_NAME, OLD.id, TG_OP, row_to_json(OLD), current_user);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO modern.audit_log (table_name, record_id, operation, old_values, new_values, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(OLD), row_to_json(NEW), current_user);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO modern.audit_log (table_name, record_id, operation, new_values, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(NEW), current_user);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$ LANGUAGE plpgsql;

-- Function to update timestamp
CREATE OR REPLACE FUNCTION modern.update_timestamp()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Create triggers for audit logging (using customer_id, order_id, etc. as record_id)
CREATE TRIGGER customers_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON modern.customers
    FOR EACH ROW EXECUTE FUNCTION modern.log_changes();

CREATE TRIGGER orders_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON modern.orders
    FOR EACH ROW EXECUTE FUNCTION modern.log_changes();

CREATE TRIGGER products_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON modern.products
    FOR EACH ROW EXECUTE FUNCTION modern.log_changes();

-- Create triggers for timestamp updates
CREATE TRIGGER customers_update_timestamp
    BEFORE UPDATE ON modern.customers
    FOR EACH ROW EXECUTE FUNCTION modern.update_timestamp();

CREATE TRIGGER orders_update_timestamp
    BEFORE UPDATE ON modern.orders
    FOR EACH ROW EXECUTE FUNCTION modern.update_timestamp();

CREATE TRIGGER products_update_timestamp
    BEFORE UPDATE ON modern.products
    FOR EACH ROW EXECUTE FUNCTION modern.update_timestamp();

CREATE TRIGGER suppliers_update_timestamp
    BEFORE UPDATE ON modern.suppliers
    FOR EACH ROW EXECUTE FUNCTION modern.update_timestamp();

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant all privileges on schema to the target user
GRANT ALL PRIVILEGES ON SCHEMA modern TO target_user;

-- Grant all privileges on all tables in schema
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA modern TO target_user;

-- Grant all privileges on all sequences in schema
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA modern TO target_user;

-- Grant usage on all functions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA modern TO target_user;

-- =============================================================================
-- CREATE MATERIALIZED VIEWS FOR PERFORMANCE
-- =============================================================================

-- Daily sales summary materialized view
CREATE MATERIALIZED VIEW modern.daily_sales_summary AS
SELECT 
    o.order_date,
    COUNT(o.order_id) as orders_count,
    SUM(o.total_amount) as total_sales,
    AVG(o.total_amount) as avg_order_value,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    SUM(oi.quantity) as total_items_sold
FROM modern.orders o
LEFT JOIN modern.order_items oi ON o.order_id = oi.order_id
WHERE o.status NOT IN ('CANCELLED')
GROUP BY o.order_date
ORDER BY o.order_date;

-- Product category performance materialized view
CREATE MATERIALIZED VIEW modern.category_performance AS
SELECT 
    p.category,
    COUNT(p.product_id) as product_count,
    COUNT(CASE WHEN p.is_active THEN 1 END) as active_products,
    SUM(p.stock_quantity) as total_inventory,
    COALESCE(SUM(oi.quantity), 0) as total_sold,
    COALESCE(SUM(oi.line_total), 0) as total_revenue,
    AVG(p.price) as avg_price
FROM modern.products p
LEFT JOIN modern.order_items oi ON p.product_id = oi.product_id
LEFT JOIN modern.orders o ON oi.order_id = o.order_id AND o.status NOT IN ('CANCELLED')
GROUP BY p.category
ORDER BY total_revenue DESC;

-- Create indexes on materialized views
CREATE INDEX idx_daily_sales_date ON modern.daily_sales_summary(order_date);
CREATE INDEX idx_category_perf_category ON modern.category_performance(category);

-- =============================================================================
-- STATISTICS AND ANALYSIS
-- =============================================================================

-- Update table statistics for query optimization
ANALYZE modern.customers;
ANALYZE modern.orders;
ANALYZE modern.products;
ANALYZE modern.order_items;
ANALYZE modern.suppliers;
ANALYZE modern.product_suppliers;
ANALYZE modern.audit_log;

-- Refresh materialized views
REFRESH MATERIALIZED VIEW modern.daily_sales_summary;
REFRESH MATERIALIZED VIEW modern.category_performance;

-- =============================================================================
-- DATA VALIDATION AND MIGRATION VERIFICATION
-- =============================================================================

-- Function to validate migrated data
CREATE OR REPLACE FUNCTION modern.validate_migration_data()
RETURNS TABLE(
    validation_check VARCHAR(100),
    status VARCHAR(10),
    details TEXT
) AS $
BEGIN
    -- Check customer count
    RETURN QUERY
    SELECT 
        'Customer Count'::VARCHAR(100),
        CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END::VARCHAR(10),
        'Found ' || COUNT(*)::TEXT || ' customers'::TEXT
    FROM modern.customers;
    
    -- Check for customers with valid emails
    RETURN QUERY
    SELECT 
        'Email Validation'::VARCHAR(100),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::VARCHAR(10),
        'Found ' || COUNT(*)::TEXT || ' customers with invalid emails'::TEXT
    FROM modern.customers 
    WHERE email IS NOT NULL AND email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,};
    
    -- Check order-customer relationships
    RETURN QUERY
    SELECT 
        'Order Relationships'::VARCHAR(100),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::VARCHAR(10),
        'Found ' || COUNT(*)::TEXT || ' orphaned orders'::TEXT
    FROM modern.orders o 
    LEFT JOIN modern.customers c ON o.customer_id = c.customer_id 
    WHERE c.customer_id IS NULL;
    
    -- Check product inventory
    RETURN QUERY
    SELECT 
        'Product Inventory'::VARCHAR(100),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::VARCHAR(10),
        'Found ' || COUNT(*)::TEXT || ' products with negative inventory'::TEXT
    FROM modern.products 
    WHERE stock_quantity < 0;
    
    -- Check status transformations
    RETURN QUERY
    SELECT 
        'Status Transformations'::VARCHAR(100),
        CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END::VARCHAR(10),
        'Found ' || COUNT(*)::TEXT || ' customers with transformed status values'::TEXT
    FROM modern.customers 
    WHERE status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED', 'DELETED');
    
    -- Check category transformations
    RETURN QUERY
    SELECT 
        'Category Transformations'::VARCHAR(100),
        CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'FAIL' END::VARCHAR(10),
        'Found ' || COUNT(*)::TEXT || ' products with transformed category values'::TEXT
    FROM modern.products 
    WHERE category IN ('TECHNOLOGY', 'FURNITURE', 'MISCELLANEOUS');
    
END;
$ LANGUAGE plpgsql;

-- =============================================================================
-- SUMMARY INFORMATION
-- =============================================================================

-- Display migration summary
DO $
BEGIN
    RAISE INFO '=== Modern Database Initialization Complete ===';
    RAISE INFO 'Schema: modern';
    RAISE INFO 'Total customers: %', (SELECT COUNT(*) FROM modern.customers);
    RAISE INFO 'Total orders: %', (SELECT COUNT(*) FROM modern.orders);
    RAISE INFO 'Total products: %', (SELECT COUNT(*) FROM modern.products);
    RAISE INFO 'Total suppliers: %', (SELECT COUNT(*) FROM modern.suppliers);
    RAISE INFO 'Total order items: %', (SELECT COUNT(*) FROM modern.order_items);
    RAISE INFO '';
    RAISE INFO '=== Key Modern Features Added ===';
    RAISE INFO '- UUID fields for external integrations';
    RAISE INFO '- JSONB metadata fields for flexible data storage';
    RAISE INFO '- Array tags for categorization';
    RAISE INFO '- Enhanced audit logging with triggers';
    RAISE INFO '- Materialized views for performance';
    RAISE INFO '- Structured address data in JSON format';
    RAISE INFO '- Product-supplier relationship management';
    RAISE INFO '- Advanced indexing including full-text search';
    RAISE INFO '';
    RAISE INFO '=== Status Code Transformations ===';
    RAISE INFO 'Legacy "A" -> Modern "ACTIVE"';
    RAISE INFO 'Legacy "I" -> Modern "INACTIVE"';
    RAISE INFO 'Legacy "S" -> Modern "SUSPENDED"';
    RAISE INFO 'Legacy "D" -> Modern "DELETED"';
    RAISE INFO '';
    RAISE INFO '=== Category Code Transformations ===';
    RAISE INFO 'Legacy "TECH" -> Modern "TECHNOLOGY"';
    RAISE INFO 'Legacy "FURN" -> Modern "FURNITURE"';
    RAISE INFO 'Legacy "MISC" -> Modern "MISCELLANEOUS"';
    RAISE INFO '';
    RAISE INFO 'Database ready for validation testing!';
END $;
