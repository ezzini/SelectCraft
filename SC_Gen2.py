import random
import sqlite3

def generate_select(conn, table_list):
    cur = conn.cursor()
    
    columns = []
    tables = []
    where_clauses = []
    join_clauses = []
    
    # get columns of each table
    for table in table_list:
        cur.execute(f"SELECT * FROM {table} LIMIT 1")
        columns.append([desc[0] for desc in cur.description])
        tables.append(table)
    
    # randomly select columns
    select_cols = []
    for i in range(random.randint(1, len(columns[0]))):
        col_index = random.randint(0, len(columns[0]) - 1)
        table_index = random.randint(0, len(tables) - 1)
        select_cols.append(tables[table_index] + "." + columns[table_index][col_index])
    
    # generate WHERE conditions
    for table in table_list:
        where_col = random.choice(columns[tables.index(table)])
        cur.execute(f"SELECT DISTINCT {where_col} FROM {table}")
        where_values = [row[0] for row in cur.fetchall()]
        where_clauses.append(f"{table}.{where_col} = '{random.choice(where_values)}'")
    
    # generate JOIN conditions
    join_table1 = random.choice(tables)
    tables.remove(join_table1)
    join_table2 = random.choice(tables)
    join_col1 = random.choice(columns[tables.index(join_table1)])
    join_col2 = random.choice(columns[tables.index(join_table2)])
    join_clauses.append(f"{join_table1} JOIN {join_table2} ON {join_table1}.{join_col1} = {join_table2}.{join_col2}")
    
    # create SELECT query
    select_query = "SELECT " + ", ".join(select_cols) + " FROM " + " ".join(join_clauses) + " WHERE " + " AND ".join(where_clauses)
    
    return select_query

# Example usage
conn = sqlite3.connect("sample.db")
table_list = ["table1", "table2", "table3"]
select_query = generate_select(conn, table_list)
print(select_query)
