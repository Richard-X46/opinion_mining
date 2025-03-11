import psycopg2
import pandas as pd
from psycopg2 import sql, extras
import os

# Load environment variables
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")
user = os.getenv("DB_USER")
password = os.getenv("PASSWORD")

def connect_to_db(host, port, database, user, password):
    try:
        conn = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )
        print("Connection successful")
        return conn
    except Exception as error:
        print(f"Error: {error}")
        return None

def create_tables(conn, create_table_query):
    """Create tables in PostgreSQL database using the provided connection and query."""
    try:
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        print("Table created or verified successfully")
    except Exception as error:
        print(f"Error: {error}")
        conn.rollback()
    finally:
        if cur is not None:
            cur.close()

def upsert_comments(conn, dataframe):
    """
    Inserts or updates comments in the database.
    """
    try:
        cur = conn.cursor()
        columns = list(dataframe.columns)
        values = [tuple(x) for x in dataframe.to_numpy()]
        
        insert_sql = sql.SQL(
            """
            INSERT INTO Comments ({}) 
            VALUES %s 
            ON CONFLICT (comment_id) DO NOTHING
            """
        ).format(sql.SQL(", ").join(map(sql.Identifier, columns)))
        
        extras.execute_values(cur, insert_sql, values)
        conn.commit()
        cur.close()
        print("Upsert operation completed successfully")
    except Exception as error:
        print(f"Error: {error}")

def upsert_table(conn, dataframe, table, primary_keys):
    """
    Inserts or updates rows in the specified table.
    
    Parameters:
        conn: psycopg2 connection.
        dataframe: Pandas DataFrame containing data to upsert.
        table: Target table name (string).
        primary_keys: List of column names that form the primary key.
    """
    try:
        cur = conn.cursor()
        columns = list(dataframe.columns)
        values = [tuple(x) for x in dataframe.to_numpy()]
        
        # Determine columns to update (exclude primary keys)
        update_columns = [col for col in columns if col not in primary_keys]
        
        # Build the ON CONFLICT update clause dynamically
        updates = sql.SQL(', ').join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
            for col in update_columns
        )
        
        insert_query = sql.SQL(
            """
            INSERT INTO {table} ({fields})
            VALUES %s
            ON CONFLICT ({pkeys}) DO UPDATE SET
            {updates}
            """
        ).format(
            table=sql.Identifier(table),
            fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
            pkeys=sql.SQL(', ').join(map(sql.Identifier, primary_keys)),
            updates=updates
        )
        
        extras.execute_values(cur, insert_query, values)
        conn.commit()
        cur.close()
        print("Upsert operation completed successfully")
    except Exception as error:
        print(f"Error: {error}")
        conn.rollback()

def query_db(conn, query, params=None):
    """
    Executes a query and returns the result as a DataFrame.
    """
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        cur.close()
        return pd.DataFrame(rows, columns=colnames)
    except Exception as error:
        print(f"Error: {error}")
        return None

# Establish connection and ensure table exists

if __name__ == "__main__":
    conn = connect_to_db(host, port, database, user, password)
    if conn:
        print("connected to postgres")
    else:
        print("failed to connect")