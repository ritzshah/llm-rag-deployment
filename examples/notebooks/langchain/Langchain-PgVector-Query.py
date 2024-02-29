#!/usr/bin/env python
# coding: utf-8

# ## Querying a Redis index
# 
# Simple example on how to query content from a PostgreSQL+pgvector VectorStore.
# 
# Requirements:
# - A PostgreSQL cluster with the pgvector extension installed (https://github.com/pgvector/pgvector)
# - A Database created in the cluster with the extension enabled (in this example, the database is named `vectordb`. Run the following command in the database as a superuser:
# `CREATE EXTENSION vector;`
# - All the information to connect to the database

# ### Needed packages

# In[1]:


get_ipython().system('pip install -q pgvector pypdf psycopg')


# ### Base parameters, the PostgreSQL info

# In[2]:


CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql-service.ic-shared-rag-llm.svc.cluster.local:5432/vectordb"
COLLECTION_NAME = "documents_test"


# ### Imports

# In[3]:


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector


# ### Initialize the connection

# In[4]:


embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings)


# ### Make a query to the index to verify sources

# In[5]:


query="How do you create a Data Science Project?"
results =store.similarity_search(query, k=4, return_metadata=True)
for result in results:
    print(result.metadata['source'])


# ### Work with a retriever

# In[6]:


retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.2 })


# In[7]:


docs = retriever.get_relevant_documents(query)
docs


# In[ ]:




