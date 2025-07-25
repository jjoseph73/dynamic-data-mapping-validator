# =============================================================================
# src/database_connector.py - Complete Database Connection Management
# Dynamic Data Mapping Validator
# =============================================================================

import os
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass

class DatabaseQueryError(Exception):
    """Raised when database query fails"""
    pass

class DatabaseConnector:
    """
    Enhanced database connector with connection pooling, retry logic,
    and comprehensive statistics collection for migration validation.
    """
    
    def __init__(self, host: str, port: int, database: str, username: str, password: str,
                 max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize database connector
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            max_retries: Maximum number of connection retries
            retry_delay: Delay between retries in seconds
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': username,
            'password': password,
            'connect_timeout': 30,
            'application_name': 'dynamic-mapping-validator'
        }
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_count = 0
        self.last_connection_time = None
        
        logger.info(f"DatabaseConnector initialized for {username}@{host}:{port}/{database}")
    
    def get_connection(self):
        """
        Get database connection with retry logic
        
        Returns:
            Database connection
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                conn = psycopg2.connect(**self.connection_params)
                self.connection_count += 1
                self.last_connection_time = datetime.now()
                logger.debug(f"Database connection successful (attempt {attempt + 1})")
                return conn
            except psycopg2.Error as e:
                last_error = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise DatabaseConnectionError(f"Failed to connect after {self.max_retries} attempts: {last_error}")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, 
                     fetch_results: bool = True) -> Optional[pd.DataFrame]:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_results: Whether to fetch and return results
            
        Returns:
            DataFrame with results or None
        """
        conn = None
        try:
            conn = self.get_connection()
            if fetch_results:
                df = pd.read_sql_query(query, conn, params=params)
                logger.debug(f"Query executed successfully, returned {len(df)} rows")
                return df
            else:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    logger.debug("Query executed successfully (no results fetched)")
                    return None
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Failed query: {query}")
            raise DatabaseQueryError(f"Query failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_table_stats(self, schema: str, table: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a table
        
        Args:
            schema: Schema name
            table: Table name
            
        Returns:
            Dictionary with table statistics
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get basic table information
            table_info = self._get_table_info(cursor, schema, table)
            
            # Get column statistics
            column_stats = self._get_column_stats(cursor, schema, table)
            
            # Get row count
            row_count = self._get_row_count(cursor, schema, table)
            
            # Get table size information
            size_info = self._get_table_size_info(cursor, schema, table)
            
            # Get constraints information
            constraints = self._get_table_constraints(cursor, schema, table)
            
            return {
                'schema': schema,
                'table': table,
                'row_count': row_count,
                'columns': column_stats,
                'table_info': table_info,
                'size_info': size_info,
                'constraints': constraints,
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get table stats for {schema}.{table}: {e}")
            raise DatabaseQueryError(f"Table stats collection failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def _get_table_info(self, cursor, schema: str, table: str) -> Dict[str, Any]:
        """Get basic table information"""
        query = """
        SELECT 
            schemaname,
            tablename,
            tableowner,
            tablespace,
            hasindexes,
            hasrules,
            hastriggers,
            rowsecurity
        FROM pg_tables 
        WHERE schemaname = %s AND tablename = %s
        """
        
        cursor.execute(query, (schema, table))
        result = cursor.fetchone()
        
        if result:
            return dict(result)
        else:
            raise DatabaseQueryError(f"Table {schema}.{table} not found")
    
    def _get_column_stats(self, cursor, schema: str, table: str) -> Dict[str, Dict[str, Any]]:
        """Get detailed column statistics"""
        # Get column information
        column_info_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            ordinal_position
        FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        
        cursor.execute(column_info_query, (schema, table))
        columns_info = cursor.fetchall()
        
        if not columns_info:
            raise DatabaseQueryError(f"No columns found for table {schema}.{table}")
        
        # Get row count for percentage calculations
        row_count = self._get_row_count(cursor, schema, table)
        
        column_stats = {}
        
        for col_info in columns_info:
            col_name = col_info['column_name']
            
            # Get null count and statistics for this column
            stats_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT({col_name}) as non_null_count,
                COUNT(*) - COUNT({col_name}) as null_count
            FROM {schema}.{table}
            """
            
            cursor.execute(stats_query)
            stats_result = cursor.fetchone()
            
            null_count = stats_result['null_count']
            non_null_count = stats_result['non_null_count']
            null_percentage = (null_count / row_count) if row_count > 0 else 0
            
            # Get additional statistics based on data type
            additional_stats = self._get_column_data_stats(cursor, schema, table, col_name, col_info['data_type'])
            
            column_stats[col_name] = {
                'data_type': col_info['data_type'],
                'is_nullable': col_info['is_nullable'],
                'column_default': col_info['column_default'],
                'character_maximum_length': col_info['character_maximum_length'],
                'numeric_precision': col_info['numeric_precision'],
                'numeric_scale': col_info['numeric_scale'],
                'ordinal_position': col_info['ordinal_position'],
                'row_count': row_count,
                'null_count': null_count,
                'non_null_count': non_null_count,
                'null_percentage': null_percentage,
                'additional_stats': additional_stats
            }
        
        return column_stats
    
    def _get_column_data_stats(self, cursor, schema: str, table: str, column: str, data_type: str) -> Dict[str, Any]:
        """Get data-type specific statistics for a column"""
        stats = {}
        
        try:
            # Numeric types
            if data_type in ['integer', 'bigint', 'smallint', 'numeric', 'decimal', 'real', 'double precision']:
                numeric_stats_query = f"""
                SELECT 
                    MIN({column}) as min_value,
                    MAX({column}) as max_value,
                    AVG({column}) as avg_value,
                    STDDEV({column}) as std_dev,
                    COUNT(DISTINCT {column}) as distinct_count
                FROM {schema}.{table}
                WHERE {column} IS NOT NULL
                """
                cursor.execute(numeric_stats_query)
                result = cursor.fetchone()
                if result:
                    # Convert decimal/numeric values to float for JSON serialization
                    for key, value in result.items():
                        if value is not None and key in ['min_value', 'max_value', 'avg_value', 'std_dev']:
                            stats[key] = float(value)
                        else:
                            stats[key] = value
            
            # String types
            elif data_type in ['character varying', 'varchar', 'text', 'character', 'char']:
                string_stats_query = f"""
                SELECT 
                    MIN(LENGTH({column})) as min_length,
                    MAX(LENGTH({column})) as max_length,
                    AVG(LENGTH({column})) as avg_length,
                    COUNT(DISTINCT {column}) as distinct_count,
                    COUNT(CASE WHEN TRIM({column}) = '' THEN 1 END) as empty_string_count
                FROM {schema}.{table}
                WHERE {column} IS NOT NULL
                """
                cursor.execute(string_stats_query)
                result = cursor.fetchone()
                if result:
                    for key, value in result.items():
                        if value is not None and key == 'avg_length':
                            stats[key] = float(value)
                        else:
                            stats[key] = value
            
            # Date/time types
            elif data_type in ['date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone']:
                date_stats_query = f"""
                SELECT 
                    MIN({column}) as min_date,
                    MAX({column}) as max_date,
                    COUNT(DISTINCT {column}) as distinct_count
                FROM {schema}.{table}
                WHERE {column} IS NOT NULL
                """
                cursor.execute(date_stats_query)
                result = cursor.fetchone()
                if result:
                    # Convert dates to strings for JSON serialization
                    for key, value in result.items():
                        if value and key in ['min_date', 'max_date']:
                            stats[key] = str(value)
                        else:
                            stats[key] = value
            
            # Boolean type
            elif data_type == 'boolean':
                boolean_stats_query = f"""
                SELECT 
                    SUM(CASE WHEN {column} = TRUE THEN 1 ELSE 0 END) as true_count,
                    SUM(CASE WHEN {column} = FALSE THEN 1 ELSE 0 END) as false_count
                FROM {schema}.{table}
                WHERE {column} IS NOT NULL
                """
                cursor.execute(boolean_stats_query)
                result = cursor.fetchone()
                if result:
                    stats.update(dict(result))
        
        except Exception as e:
            logger.warning(f"Failed to get detailed stats for column {column}: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _get_row_count(self, cursor, schema: str, table: str) -> int:
        """Get exact row count for table"""
        count_query = f"SELECT COUNT(*) as row_count FROM {schema}.{table}"
        cursor.execute(count_query)
        result = cursor.fetchone()
        return result['row_count'] if result else 0
    
    def _get_table_size_info(self, cursor, schema: str, table: str) -> Dict[str, Any]:
        """Get table size and storage information"""
        size_query = """
        SELECT 
            pg_size_pretty(pg_total_relation_size(%s)) as total_size,
            pg_size_pretty(pg_relation_size(%s)) as table_size,
            pg_size_pretty(pg_total_relation_size(%s) - pg_relation_size(%s)) as index_size,
            pg_total_relation_size(%s) as total_size_bytes,
            pg_relation_size(%s) as table_size_bytes
        """
        
        full_table_name = f"{schema}.{table}"
        cursor.execute(size_query, (full_table_name,) * 6)
        result = cursor.fetchone()
        
        return dict(result) if result else {}
    
    def _get_table_constraints(self, cursor, schema: str, table: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get table constraints information"""
        constraints_query = """
        SELECT 
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        LEFT JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = %s AND tc.table_name = %s
        ORDER BY tc.constraint_type, tc.constraint_name
        """
        
        cursor.execute(constraints_query, (schema, table))
        constraints_raw = cursor.fetchall()
        
        # Group constraints by type
        constraints = {
            'primary_key': [],
            'foreign_key': [],
            'unique': [],
            'check': []
        }
        
        for constraint in constraints_raw:
            constraint_dict = dict(constraint)
            constraint_type = constraint_dict['constraint_type'].lower()
            
            if constraint_type == 'primary key':
                constraints['primary_key'].append(constraint_dict)
            elif constraint_type == 'foreign key':
                constraints['foreign_key'].append(constraint_dict)
            elif constraint_type == 'unique':
                constraints['unique'].append(constraint_dict)
            elif constraint_type == 'check':
                constraints['check'].append(constraint_dict)
        
        return constraints
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test database connection and return connection info
        
        Returns:
            Dictionary with connection test results
        """
        conn = None
        try:
            start_time = time.time()
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get database version and basic info
            cursor.execute("SELECT version(), current_database(), current_user, current_timestamp")
            result = cursor.fetchone()
            
            connection_time = time.time() - start_time
            
            return {
                'status': 'success',
                'connection_time_ms': round(connection_time * 1000, 2),
                'database_version': result[0],
                'database_name': result[1],
                'connected_user': result[2],
                'server_time': str(result[3]),
                'connection_params': {
                    'host': self.connection_params['host'],
                    'port': self.connection_params['port'],
                    'database': self.connection_params['database'],
                    'user': self.connection_params['user']
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'connection_params': {
                    'host': self.connection_params['host'],
                    'port': self.connection_params['port'],
                    'database': self.connection_params['database'],
                    'user': self.connection_params['user']
                }
            }
        finally:
            if conn:
                conn.close()
    
    def get_schema_tables(self, schema: str) -> List[str]:
        """
        Get list of tables in a schema
        
        Args:
            schema: Schema name
            
        Returns:
            List of table names
        """
        query = """
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = %s 
        ORDER BY tablename
        """
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (schema,))
            results = cursor.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Failed to get tables for schema {schema}: {e}")
            raise DatabaseQueryError(f"Schema tables query failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_all_schemas(self) -> List[str]:
        """
        Get list of all schemas in the database
        
        Returns:
            List of schema names
        """
        query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast', 'pg_temp_1', 'pg_toast_temp_1')
        ORDER BY schema_name
        """
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Failed to get schemas: {e}")
            raise DatabaseQueryError(f"Schemas query failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information and statistics
        
        Returns:
            Dictionary with connection information
        """
        return {
            'connection_params': {
                'host': self.connection_params['host'],
                'port': self.connection_params['port'],
                'database': self.connection_params['database'],
                'user': self.connection_params['user']
            },
            'connection_count': self.connection_count,
            'last_connection_time': self.last_connection_time.isoformat() if self.last_connection_time else None,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }
    
    def __str__(self) -> str:
        """String representation of the connector"""
        return f"DatabaseConnector({self.connection_params['user']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"DatabaseConnector(host='{self.connection_params['host']}', port={self.connection_params['port']}, database='{self.connection_params['database']}', user='{self.connection_params['user']}')"

# Utility functions
def create_connector_from_env(prefix: str) -> DatabaseConnector:
    """
    Create database connector from environment variables
    
    Args:
        prefix: Environment variable prefix (e.g., 'SOURCE_DB' or 'TARGET_DB')
        
    Returns:
        DatabaseConnector instance
    """
    return DatabaseConnector(
        host=os.getenv(f'{prefix}_HOST', 'localhost'),
        port=int(os.getenv(f'{prefix}_PORT', 5432)),
        database=os.getenv(f'{prefix}_NAME', 'postgres'),
        username=os.getenv(f'{prefix}_USER', 'postgres'),
        password=os.getenv(f'{prefix}_PASSWORD', 'postgres')
    )

def test_database_connectivity(connectors: List[DatabaseConnector]) -> Dict[str, Dict[str, Any]]:
    """
    Test connectivity for multiple database connectors
    
    Args:
        connectors: List of DatabaseConnector instances
        
    Returns:
        Dictionary with test results for each connector
    """
    results = {}
    
    for i, connector in enumerate(connectors):
        connector_name = f"database_{i + 1}"
        try:
            test_result = connector.test_connection()
            results[connector_name] = test_result
        except Exception as e:
            results[connector_name] = {
                'status': 'error',
                'error': str(e),
                'connector_info': str(connector)
            }
    
    return results

# Export main classes and functions
__all__ = [
    'DatabaseConnector',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'create_connector_from_env',
    'test_database_connectivity'
]
