# LLM RAG Deployment Helm Charts

This directory contains Helm charts converted from the original `bootstrap` and `bootstrap-rag` Kustomize deployments.

## Charts

### unified-bootstrap-chart ⭐ **RECOMMENDED**
A unified chart combining both bootstrap and bootstrap-rag components into a single deployment:
- All core infrastructure components
- All RAG-specific components
- Shared Gradio configuration
- Independent component control
- Single values.yaml and Chart.yaml

### bootstrap-chart (Legacy)
Core infrastructure components for the LLM RAG deployment including:
- RHOAI (Red Hat OpenShift AI) setup
- Shared services (MinIO, Database, Milvus)
- LLM serving infrastructure
- Basic UI components

### bootstrap-rag-chart (Legacy)  
RAG-specific components including:
- RAG-optimized storage and database configurations
- Data science pipelines
- RAG-specific LLM serving
- Vector database setup for embeddings

## Quick Start

### Option 1: Unified Chart (Recommended)
```bash
cd helm-chart-new/unified-bootstrap-chart
helm install unified-bootstrap . -n openshift-gitops
```

### Option 2: Separate Charts (Legacy)
1. **Install core components first:**
   ```bash
   cd helm-chart-new/bootstrap-chart
   helm install bootstrap-deployment . -n openshift-gitops
   ```

2. **Then install RAG components:**
   ```bash
   cd ../bootstrap-rag-chart
   helm install bootstrap-rag-deployment . -n openshift-gitops
   ```

## Configuration

Both charts use ArgoCD ApplicationSets to manage deployments. Key configuration options:

- `global.repoURL`: Git repository containing the manifests
- `global.targetRevision`: Git branch/tag to deploy from
- `components[].enabled`: Enable/disable individual components

## Prerequisites

- OpenShift cluster
- ArgoCD installed in `openshift-gitops` namespace
- Sufficient resources for AI/ML workloads

## Customization

Create custom values files for environment-specific configurations:

```yaml
# values-dev.yaml
global:
  targetRevision: develop

components:
  - name: gradio
    enabled: false  # Disable in dev
```

```bash
helm install bootstrap-deployment ./bootstrap-chart -f values-dev.yaml
```

## Dependencies

The bootstrap-rag-chart depends on components from bootstrap-chart being deployed first, particularly:
- RHOAI operator and installation
- Shared MinIO and database services
- Base networking and security configurations

## Monitoring

Monitor deployments through ArgoCD UI or CLI:
```bash
kubectl get applications -n openshift-gitops
kubectl get applicationsets -n openshift-gitops
```

## Original Structure

These charts were converted from the original directory structure:
- `bootstrap/` → `bootstrap-chart/`  
- `bootstrap-rag/` → `bootstrap-rag-chart/`

The original Kustomize configurations are preserved within the chart templates while adding Helm templating capabilities for better configuration management.