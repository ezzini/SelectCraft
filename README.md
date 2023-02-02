[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/open-source.svg)](https://forthebadge.com)

# SelectCraft - The SQL Generation Script
An SQL generation script that allows you to generate different SELECT statements based on the structure of a given SQLite database. 

## Features
- Generate SELECT statements with up to X columns and Y conditions
- Supports different types of WHERE conditions, including IN, BETWEEN, and comparison operators
- Automatically determines the type of columns in the database and applies the appropriate conditions
- Includes JOIN conditions in the generated SELECT statements

## Usage
1. Connect the script to your SQLite database
2. Call the `generate_select` function, passing in the database connection and the name of the table you want to query.
3. The function will return a SELECT statement, ready to be executed against the database.

## Example
```python
import sqlite3
from SelectCraft import generate_select

conn = sqlite3.connect('example.db')
select_statement = generate_select(conn, 'table_name')
print(select_statement)
```

## Requirements
- Python 3
- SQLite3

# Contributions
Contributions are always welcome. If you have an idea for a feature or have found a bug, please open an issue on the GitHub repository.

# License
SelectCraft is licensed under the MIT License.
