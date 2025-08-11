# Bootstrap Chart

This Helm chart deploys the LLM RAG Bootstrap components using ArgoCD ApplicationSets.

## Overview

The bootstrap chart includes the core infrastructure components needed for the LLM RAG deployment:

- RHOAI Operator
- RHOAI Installation & Configuration  
- Shared MinIO for object storage
- Shared Database
- Milvus vector database
- LLM serving infrastructure
- PgVector deployment
- Gradio UI
- Image detection services
- User project management
- Application deployment

## Installation

```bash
# Install the chart
helm install bootstrap-deployment ./bootstrap-chart

# Install with custom values
helm install bootstrap-deployment ./bootstrap-chart -f custom-values.yaml

# Upgrade
helm upgrade bootstrap-deployment ./bootstrap-chart
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
  - name: ic-rhoai-operator
    path: bootstrap/ic-rhoai-operator
    enabled: true
```

### Enabling/Disabling Components

To disable specific components, set `enabled: false`:

```yaml
components:
  - name: gradio
    path: bootstrap/gradio
    enabled: false  # Disable Gradio UI
```

### Custom Repository

To use a different repository:

```yaml
global:
  repoURL: https://github.com/your-org/your-repo.git
  targetRevision: main
```

## Components

| Component | Description | Default |
|-----------|-------------|---------|
| ic-rhoai-operator | Red Hat OpenShift AI Operator | Enabled |
| ic-rhoai-installation | RHOAI Installation | Enabled |
| ic-rhoai-configuration | RHOAI Configuration | Enabled |
| ic-shared-minio-app | MinIO Object Storage | Enabled |
| ic-shared-database-app | PostgreSQL Database | Enabled |
| ic-shared-milvus | Milvus Vector Database | Enabled |
| ic-shared-llm-app | LLM Serving Infrastructure | Enabled |
| pg-vector-deployment | PgVector Extension | Enabled |
| gradio | Gradio Web UI | Enabled |
| ic-shared-img-det | Image Detection Services | Enabled |
| ic-user-projects | User Project Management | Enabled |
| ic-shared-app | Main Application | Enabled |

## Prerequisites

- OpenShift cluster with ArgoCD installed
- Sufficient cluster resources for AI/ML workloads
- Access to container registries for required images

## Troubleshooting

Check ArgoCD applications:
```bash
kubectl get applications -n openshift-gitops
kubectl describe application <app-name> -n openshift-gitops
```

View ApplicationSet:
```bash
kubectl get applicationsets -n openshift-gitops
kubectl describe applicationset bootstrap -n openshift-gitops
```