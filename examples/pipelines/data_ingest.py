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

# Dependencies are managed via requirements.txt
# Install with: pip install -r requirements.txt


# ### Auto-fetch latest OpenShift AI version

# In[2]:


import requests
import re
import os
from bs4 import BeautifulSoup


# Ordered list of versions to try — newest first. Add new versions here as they release.
KNOWN_VERSIONS = ["3.2", "3.1", "3.0", "2.22", "2.21", "2.20", "2.19", "2.18", "2.17", "2.16"]

PDF_BASE_URL = "https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed"

DOCUMENTS = [
    "release_notes",
    "introduction_to_red_hat_openshift_ai",
    "getting_started_with_red_hat_openshift_ai_self-managed",
]


def pdf_url(version, doc):
    return f"{PDF_BASE_URL}/{version}/pdf/{doc}/red_hat_openshift_ai_self-managed-{version}-{doc}-en-us.pdf"


def html_url(version, doc):
    return f"{PDF_BASE_URL}/{version}/html-single/{doc}/index"


def check_pdf_available(version, doc):
    """Check if a PDF is actually downloadable for a given version and document."""
    url = pdf_url(version, doc)
    try:
        resp = requests.head(url, allow_redirects=True, timeout=15)
        return resp.status_code == 200
    except Exception:
        return False


def find_working_version():
    """
    Try known versions in order and return the first one where at least one PDF is available.
    This avoids relying on scraping the docs landing page which can change structure.
    """
    for version in KNOWN_VERSIONS:
        print(f"Checking version {version}...")
        if check_pdf_available(version, DOCUMENTS[0]):
            print(f"Version {version} has available PDFs")
            return version
        else:
            print(f"Version {version} PDFs not available, trying next...")

    print(f"No working version found, falling back to {KNOWN_VERSIONS[0]}")
    return KNOWN_VERSIONS[0]


product_version = find_working_version()
print(f"Using Red Hat OpenShift AI Self-Managed version: {product_version}")


# ### Base parameters, the PostgreSQL info

# In[3]:

CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql-service.ic-shared-rag-llm.svc.cluster.local:5432/vectordb"
COLLECTION_NAME = "documents_test"


# #### Imports

# In[4]:


from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector


# ## Initial index creation and document ingestion

# #### Download and load pdfs

# In[5]:


pdfs = [pdf_url(product_version, doc) for doc in DOCUMENTS]
pdfs_to_urls = {
    f"red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us": html_url(product_version, doc)
    for doc in DOCUMENTS
}


# In[6]:


pdf_dir = f"rhoai-doc-{product_version}"
os.makedirs(pdf_dir, exist_ok=True)

downloaded_count = 0
for pdf in pdfs:
    try:
        response = requests.get(pdf, timeout=60)
    except Exception as e:
        print(f"Skipped {pdf} - error: {e}")
        continue
    if response.status_code != 200:
        print(f"Skipped {pdf} - status {response.status_code}")
        continue
    filename = pdf.split('/')[-1]
    with open(f"{pdf_dir}/{filename}", 'wb') as f:
        f.write(response.content)
    downloaded_count += 1
    print(f"Downloaded {filename}")

if downloaded_count == 0:
    print("WARNING: No PDFs were downloaded. Will proceed with website content only.")


# In[7]:


pdf_folder_path = f"./{pdf_dir}"
pdf_docs = []

if downloaded_count > 0:
    pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
    pdf_docs = pdf_loader.load()
    print(f"Loaded {len(pdf_docs)} pages from PDFs")
else:
    print("No PDFs to load, skipping PDF ingestion")


# #### Inject metadata

# In[8]:


from pathlib import Path

for doc in pdf_docs:
    stem = Path(doc.metadata["source"]).stem
    if stem in pdfs_to_urls:
        doc.metadata["source"] = pdfs_to_urls[stem]


# #### Load websites

# In[9]:


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


# In[10]:


website_docs = []
try:
    website_loader = WebBaseLoader(websites)
    website_docs = website_loader.load()
    print(f"Loaded {len(website_docs)} website documents")
except Exception as e:
    print(f"WARNING: Failed to load some websites: {e}")


# #### Merge both types of docs

# In[11]:


docs = pdf_docs + website_docs

if len(docs) == 0:
    print("ERROR: No documents were loaded at all. Exiting.")
    exit(1)

print(f"Total documents to ingest: {len(docs)}")


# #### Split documents into chunks with some overlap

# In[12]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)
print(f"Total chunks after splitting: {len(all_splits)}")


# #### Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.

# In[13]:


for doc in all_splits:
    doc.page_content = doc.page_content.replace('\x00', '')


# #### Create the index and ingest the documents

# In[14]:


embeddings = HuggingFaceEmbeddings()

db = PGVector.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

print("Successfully ingested documents into PGVector")


# #### Test query

# In[16]:


query = "What is the latest version of Red Hat openshift AI self managed?"
docs_with_score = db.similarity_search_with_score(query)


# In[17]:


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
