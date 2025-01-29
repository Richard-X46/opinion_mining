import sql3

my_query = "SELECT * FROM Comments"

df = sql3.db_fetch_as_frame(db_path="comments_database.sqlite", query=my_query)

# sql3.list_tables("comments_database.sqlite")

print(df)