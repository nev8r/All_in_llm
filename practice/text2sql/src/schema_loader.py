"""
schema_loader.py
=================
功能：
- 连接数据库（SQLite）
- 获取所有表名与字段信息
- 输出 schema 描述（字典或字符串形式）
"""

import sqlite3
from typing import Dict, List

def get_db_schema(db_path: str) -> Dict[str, List[str]]:
    """返回数据库的表结构信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [t[0] for t in cursor.fetchall()]

    for table in table_names:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [col[1] for col in cursor.fetchall()]
        tables[table] = columns

    conn.close()
    return tables


def format_schema_for_prompt(schema: Dict[str, List[str]]) -> str:
    """将 schema 转为可读字符串（用于 prompt）"""
    lines = []
    for table, cols in schema.items():
        lines.append(f"表 {table}: " + ", ".join(cols))
    return "\n".join(lines)


if __name__ == "__main__":
    schema = get_db_schema("db/sample.db")
    print("数据库结构：")
    print(format_schema_for_prompt(schema))
