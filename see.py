import sqlite3
import os

conn = sqlite3.connect('photo_metadata.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM metadata')
# cursor.execute('drop table metadata')

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()


# if os.path.exists("faiss.index"):
#     os.remove("faiss.index")

# if os.path.exists("photo_metadata.db"):
#     os.remove("photo_metadata.db")

# if os.path.exists("faiss_map.json"):
#     os.remove("faiss_map.json")

