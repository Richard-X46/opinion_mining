import psycopg2
import pandas as pd
from psycopg2 import sql, extras
import os
import sqlite3
# load environment variables


host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")
user = os.getenv("USER")
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


def generate_create_table_sql(
    table_name: str, dataframe: pd.DataFrame, primary_keys: list
) -> str:
    columns = dataframe.columns
    column_definitions = []
    for column in columns:
        if column in primary_keys:
            column_definitions.append(f"{column} VARCHAR PRIMARY KEY")
        else:
            column_definitions.append(f"{column} VARCHAR")
    create_table_sql = (
        f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_definitions)});"
    )
    return create_table_sql


def create_table(conn, table_name, dataframe, primary_keys):
    create_table_sql = generate_create_table_sql(table_name, dataframe, primary_keys)
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        print("Table created successfully")
    except Exception as error:
        print(f"Error: {error}")


def upsert_table(conn, table_name, dataframe):
    try:
        cur = conn.cursor()
        columns = list(dataframe.columns)
        values = [tuple(x) for x in dataframe.to_numpy()]

        insert_sql = sql.SQL(
            "INSERT INTO {} ({}) VALUES %s ON CONFLICT (id) DO UPDATE SET {}"
        ).format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(
                [
                    sql.SQL("{} = EXCLUDED.{}").format(
                        sql.Identifier(col), sql.Identifier(col)
                    )
                    for col in columns
                ]
            ),
        )

        extras.execute_values(cur, insert_sql, values)
        conn.commit()
        cur.close()
        print("Upsert operation completed successfully")
    except Exception as error:
        print(f"Error: {error}")


def query_db(conn, query):
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        cur.close()
        return pd.DataFrame(rows, columns=colnames)
    except Exception as error:
        print(f"Error: {error}")
        return None


# Example usage

def db_fetch_as_frame(db_path: str, query: str) -> pd.DataFrame:
    """
    Read data from a SQLite database.

    Args:
    db_path (str): Path to the SQLite database file
    table_name (str): Name of the table to be read

    Returns:
    pd.DataFrame: DataFrame containing the data from the table
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


conn = connect_to_db(host, port, database, user, password)


query = """ select * from Comments """

df = db_fetch_as_frame("comments_database.sqlite", query)





