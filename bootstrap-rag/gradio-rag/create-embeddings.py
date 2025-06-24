from sentence_transformers import SentenceTransformer
import psycopg2

#podman run -d \
#  --name postgres-vectordb \
#  -e POSTGRES_USER=vectordb_user \
#  -e POSTGRES_PASSWORD=yourpassword \
#  -e POSTGRES_DB=vectordb \
#  -p 5432:5432 \
#  pgvector/pgvector:pg16

model = SentenceTransformer('all-MiniLM-L6-v2')
text = "Ritesh Shah is from Red Hat and is a Senior Principal Architect focused on AI/ML related products and technologies. He speaks at various forums and is well established in the IT industry as a veteran with over 20 years of experience in open source technologies and cloud computing."
embedding = model.encode(text).tolist()
embedding_str = str(embedding)  # '[0.123, 0.456, ...]'

conn = psycopg2.connect(
    dbname="vectordb", user="vectordb_user", password="newpasswd", host="0.0.0.0", port=5432
)
cur = conn.cursor()
insert_sql = "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)"
cur.execute(insert_sql, (text, embedding_str))
conn.commit()
cur.close()
conn.close()
print("Inserted successfully!")
