# Bootstrap RAG Chart

This Helm chart deploys the LLM RAG Bootstrap RAG-specific components using ArgoCD ApplicationSets.

## Overview

The bootstrap-rag chart includes specialized RAG (Retrieval-Augmented Generation) components:

- RAG-specific MinIO configuration
- RAG Database setup
- RAG LLM serving
- Gradio RAG interface
- PgVector RAG deployment
- Data Science RAG pipelines
- RAG-specific configurations

## Installation

```bash
# Install the chart
helm install bootstrap-rag-deployment ./bootstrap-rag-chart

# Install with custom values
helm install bootstrap-rag-deployment ./bootstrap-rag-chart -f custom-values.yaml

# Upgrade
helm upgrade bootstrap-rag-deployment ./bootstrap-rag-chart
```

## Configuration

### Basic Configuration

Key configuration options in `values.yaml`:

```yaml
global:
  repoURL: https://github.com/ritzshah/llm-rag-deployment.git
  targetRevision: update-token-v1
  cluster: in-cluster

components:
  - name: shared-rag-minio-app
    path: bootstrap-rag/shared-rag-minio
    enabled: true
```

### Enabling/Disabling Components

To disable specific components, set `enabled: false`:

```yaml
components:
  - name: workbench
    path: bootstrap-rag/notebook
    enabled: false  # Workbench disabled by default
```

### RAG-Specific Components

Additional RAG components can be configured:

```yaml
ragComponents:
  rhoaiConfiguration:
    enabled: true
    path: bootstrap-rag/rhoai-rag-configuration
  secretCreate:
    enabled: true
    path: bootstrap-rag/secret-create
```

## Components

| Component | Description | Default |
|-----------|-------------|---------|
| shared-rag-minio-app | RAG MinIO Storage | Enabled |
| shared-rag-database-app | RAG Database | Enabled |
| shared-rag-llm-app | RAG LLM Services | Enabled |
| gradio-rag-app | RAG Gradio Interface | Enabled |
| pgvector-rag-deployment-app | RAG PgVector Setup | Enabled |
| ds-rag-pipeline | Data Science RAG Pipelines | Enabled |
| workbench | Development Workbench | Disabled |

### Additional RAG Components

| Component | Description | Default |
|-----------|-------------|---------|
| rhoaiConfiguration | RHOAI RAG Configuration | Enabled |
| secretCreate | Secret Management | Enabled |
| odfS3 | Object Storage Configuration | Enabled |
| modelServer | Model Serving Setup | Enabled |
| workbenchImage | Custom Workbench Images | Enabled |

## Prerequisites

- OpenShift cluster with ArgoCD installed
- Bootstrap components installed (from bootstrap-chart)
- Sufficient cluster resources for RAG workloads
- Vector database support

## Data Science Pipelines

The RAG chart includes data science pipeline components for:
- Data ingestion and validation
- Embedding generation
- Model training and evaluation
- Pipeline orchestration

## Troubleshooting

Check ArgoCD applications:
```bash
kubectl get applications -n openshift-gitops | grep rag
kubectl describe application <rag-app-name> -n openshift-gitops
```

View RAG ApplicationSet:
```bash
kubectl get applicationsets -n openshift-gitops
kubectl describe applicationset bootstrap-rag -n openshift-gitops
```

Check pipeline status:
```bash
kubectl get pods -n <pipeline-namespace>
kubectl logs <pipeline-pod> -n <pipeline-namespace>
```