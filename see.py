import sqlite3
import os

conn = sqlite3.connect('photo_metadata.db')
cursor = conn.cursor()

# cursor.execute('SELECT * FROM metadata')
cursor.execute('drop table metadata')

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()


if os.path.exists("image_embeddings.npz"):
    os.remove("image_embeddings.npz")