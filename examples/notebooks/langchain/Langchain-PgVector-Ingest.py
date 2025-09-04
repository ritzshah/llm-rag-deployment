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


get_ipython().system('pip install pgvector pypdf psycopg langchain langchain-community beautifulsoup4 sentence-transformers')


# ### Auto-fetch latest OpenShift AI version

# In[2]:


import requests
import re
from bs4 import BeautifulSoup

def get_latest_openshift_ai_version():
    """
    Fetch the latest Red Hat OpenShift AI Self-Managed version from the documentation page.
    Returns the version string like "2.22", "3.1", etc.
    """
    try:
        # First try the original URL, handle redirect if needed
        url = "https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, allow_redirects=True)
        
        # If redirected, use the final URL
        if response.history:
            print(f"Redirected to: {response.url}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for version patterns specifically for RHOAI versions
            # More specific patterns to avoid false matches
            version_patterns = [
                r'/red_hat_openshift_ai_self-managed/(\d+\.\d+)/',  # In URLs like /red_hat_openshift_ai_self-managed/2.22/
                r'red_hat_openshift_ai_self-managed/(\d+\.\d+)',   # In paths
                r'rhoai[_-](\d+\.\d+)',  # "rhoai_2.22" or "rhoai-2.22"
                r'version[_\s-]*(\d+\.\d+)',  # "version 2.22" or "version-2.22"
            ]
            
            # Search in page text and URLs
            page_text = soup.get_text()
            versions_found = []
            
            # Search in URLs and links first (more reliable)
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                for pattern in version_patterns:
                    matches = re.findall(pattern, href, re.IGNORECASE)
                    versions_found.extend(matches)
            
            # Also search in page text
            for pattern in version_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                versions_found.extend(matches)
            
            # Filter versions to only include valid RHOAI versions (X.Y format where X >= 2)
            valid_versions = []
            for version in versions_found:
                # Check if it's a valid RHOAI version format (X.Y where X is major version >= 2)
                if re.match(r'^\d+\.\d{1,2}$', version):
                    # Additional validation: major version should be >= 2, minor should be reasonable
                    major, minor = version.split('.')
                    if int(major) >= 2 and 0 <= int(minor) <= 99:
                        # Filter out obviously wrong versions like 192.x
                        if int(major) <= 10:  # Reasonable upper bound for major version
                            valid_versions.append(version)
            
            # Remove duplicates and sort to get the latest
            unique_versions = list(set(valid_versions))
            if unique_versions:
                # Sort versions (assuming format X.Y)
                sorted_versions = sorted(unique_versions, key=lambda x: [int(i) for i in x.split('.')], reverse=True)
                latest_version = sorted_versions[0]
                print(f"Found valid versions: {sorted_versions}")
                print(f"Using latest version: {latest_version}")
                return latest_version
            else:
                print("No valid version found in the documentation page")
                return "2.22"  # Fallback to a recent known version
        else:
            print(f"Failed to fetch documentation page. Status code: {response.status_code}")
            return "2.22"  # Fallback
            
    except Exception as e:
        print(f"Error fetching version: {e}")
        return "2.22"  # Fallback to a recent known version

# Get the latest version dynamically
product_version = get_latest_openshift_ai_version()
print(f"Using Red Hat OpenShift AI Self-Managed version: {product_version}")


# ### Base parameters, the PostgreSQL info

# In[3]:


CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql-service.ic-shared-rag-llm.svc.cluster.local:5432/vectordb"
COLLECTION_NAME = "documents_test"


# #### Imports

# In[4]:


from langchain.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector


# ## Initial index creation and document ingestion

# #### Download and load pdfs

# In[5]:


documents = [
    "release_notes",
    "introduction_to_red_hat_openshift_ai",
    "getting_started_with_red_hat_openshift_ai_self-managed",   
]

pdfs = [f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/pdf/{doc}/red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us.pdf" for doc in documents]
pdfs_to_urls = {f"red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us": f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/html-single/{doc}/index" for doc in documents}


# In[6]:


import os

os.makedirs(f"rhoai-doc-{product_version}", exist_ok=True)

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


# In[7]:


pdf_folder_path = f"./rhoai-doc-{product_version}"

pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
pdf_docs = pdf_loader.load()


# #### Inject metadata

# In[8]:


from pathlib import Path

for doc in pdf_docs:
    doc.metadata["source"] = pdfs_to_urls[Path(doc.metadata["source"]).stem]


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


website_loader = WebBaseLoader(websites)
website_docs = website_loader.load()


# #### Merge both types of docs

# In[11]:


docs = pdf_docs + website_docs


# #### Split documents into chunks with some overlap

# In[12]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)
all_splits[0]


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
    #pre_delete_collection=True # This deletes existing collection and its data, use carefully!
)


# #### Alternatively, add new documents

# In[15]:


# embeddings = HuggingFaceEmbeddings()

# db = PGVector(
#     connection_string=CONNECTION_STRING,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings)

# db.add_documents(all_splits)


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


# In[ ]:




