#!/usr/bin/env python
# coding: utf-8

# ## Creating an index and populating it with documents using PostgreSQL+pgvector
# 
# Simple example on how to ingest PDF documents, then web pages content into a PostgreSQL+pgvector VectorStore.
# 
# Requirements:
# - A PostgreSQL cluster with the pgvector extension installed (https://github.com/pgvector/pgvector)
# - A Database created in the cluster with the extension enabled (in this example, the database is named `vectordb`. Run the following command in the database as a superuser:
# `CREATE EXTENSION vector;`
# 
# Note: if your PostgreSQL is deployed on OpenShift, directly from inside the Pod (Terminal view on the Console, or using `oc rsh` to log into the Pod), you can run the command: `psql -d vectordb -c "CREATE EXTENSION vector;"`
# 

# ### Needed packages

# In[1]:


#get_ipython().system('pip install pgvector pypdf psycopg')


# ### Base parameters, the PostgreSQL info

# In[2]:


product_version = "2-latest"
CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql-service.ic-shared-rag-llm.svc.cluster.local:5432/vectordb"
COLLECTION_NAME = "documents_test"


# #### Imports

# In[3]:


from langchain.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector


# ## Initial index creation and document ingestion

# #### Download and load pdfs

# In[4]:


documents = [
    "release_notes",
    "introduction_to_red_hat_openshift_ai",
    "getting_started_with_red_hat_openshift_ai_self-managed",   
]

pdfs = [f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/pdf/{doc}/red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us.pdf" for doc in documents]
pdfs_to_urls = {f"red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us": f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/html-single/{doc}/index" for doc in documents}


# In[5]:


import requests
import os

os.mkdir(f"rhoai-doc-{product_version}")

for pdf in pdfs:
    try:
        response = requests.get(pdf)
    except:
        print(f"Skipped {pdf}")
        continue
    if response.status_code!=200:
        print(f"Skipped {pdf}")
        continue  
    with open(f"rhoai-doc-{product_version}/{pdf.split('/')[-1]}", 'wb') as f:
        f.write(response.content)


# In[6]:


pdf_folder_path = f"./rhoai-doc-{product_version}"

pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
pdf_docs = pdf_loader.load()


# #### Inject metadata

# In[7]:


from pathlib import Path

for doc in pdf_docs:
    doc.metadata["source"] = pdfs_to_urls[Path(doc.metadata["source"]).stem]


# #### Load websites

# In[8]:


websites = [
    "https://ai-on-openshift.io/getting-started/openshift/",
    "https://ai-on-openshift.io/getting-started/opendatahub/",
    "https://ai-on-openshift.io/getting-started/openshift-ai/",
    "https://ai-on-openshift.io/odh-rhoai/configuration/",
    "https://ai-on-openshift.io/odh-rhoai/custom-notebooks/",
    "https://ai-on-openshift.io/odh-rhoai/nvidia-gpus/",
    "https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton/",
    "https://ai-on-openshift.io/odh-rhoai/openshift-group-management/",
    "https://ai-on-openshift.io/tools-and-applications/minio/minio/",
    "https://access.redhat.com/articles/7047935",
    "https://access.redhat.com/articles/rhoai-supported-configs",
]


# In[9]:


website_loader = WebBaseLoader(websites)
website_docs = website_loader.load()


# #### Merge both types of docs

# In[10]:


docs = pdf_docs + website_docs


# #### Split documents into chunks with some overlap

# In[11]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)
all_splits[0]


# #### Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.

# In[12]:


for doc in all_splits:
    doc.page_content = doc.page_content.replace('\x00', '')


# #### Create the index and ingest the documents

# In[13]:


embeddings = HuggingFaceEmbeddings()

db = PGVector.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    #pre_delete_collection=True # This deletes existing collection and its data, use carefully!
)


# #### Alternatively, add new documents

# In[14]:


# embeddings = HuggingFaceEmbeddings()

# db = PGVector(
#     connection_string=CONNECTION_STRING,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings)

# db.add_documents(all_splits)


# #### Test query

# In[15]:


query = "What is the latest version of Red Hat openshift AI self managed?"
docs_with_score = db.similarity_search_with_score(query)


# In[16]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

# In[ ]:
