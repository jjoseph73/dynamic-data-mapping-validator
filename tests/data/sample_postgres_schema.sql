# tests/data/sample_postgres_schema.sql
-- Corresponding PostgreSQL schema
CREATE SCHEMA IF NOT EXISTS public;

CREATE TABLE public.employees (
    employee_id INTEGER NOT NULL,
    first_name VARCHAR(20),
    last_name VARCHAR(25) NOT NULL,
    email VARCHAR(25) NOT NULL,
    phone VARCHAR(20),
    hire_date DATE NOT NULL,
    job_id VARCHAR(10) NOT NULL,
    salary DECIMAL(8,2),
    commission_rate DECIMAL(3,2),
    manager_id INTEGER,
    department_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT pk_employees PRIMARY KEY (employee_id),
    CONSTRAINT uk_employees_email UNIQUE (email)
);

CREATE TABLE public.departments (
    department_id INTEGER NOT NULL,
    department_name VARCHAR(30) NOT NULL,
    manager_id INTEGER,
    location_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT pk_departments PRIMARY KEY (department_id)
);

CREATE TABLE public.customers (
    customer_id BIGINT NOT NULL,
    company_name VARCHAR(100),
    contact_name VARCHAR(50),
    contact_title VARCHAR(30),
    address VARCHAR(60),
    city VARCHAR(15),
    region VARCHAR(15),
    postal_code VARCHAR(10),
    country VARCHAR(15),
    phone VARCHAR(24),
    fax VARCHAR(24),
    email VARCHAR(255),
    credit_limit DECIMAL(10,2),
    account_manager_id INTEGER,
    customer_type_id SMALLINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT pk_customers PRIMARY KEY (customer_id)
);