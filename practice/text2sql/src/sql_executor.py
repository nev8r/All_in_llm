"""
sql_executor.py
================
功能：
- 执行输入的 SQL 查询
- 返回结果 DataFrame 或打印表格
"""

import sqlite3
import pandas as pd

def execute_sql(db_path: str, query: str):
    """执行 SQL 并返回 DataFrame"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        print(f"✅ SQL 执行成功:\n{query}\n")
        print(df)
        return df
    except Exception as e:
        print(f"❌ SQL 执行错误: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    test_query = "SELECT * FROM users LIMIT 3;"
    execute_sql("db/sample.db", test_query)
