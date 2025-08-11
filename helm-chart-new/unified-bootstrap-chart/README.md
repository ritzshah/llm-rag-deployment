# Unified Bootstrap Chart

This unified Helm chart combines both the Bootstrap and Bootstrap-RAG components into a single, manageable deployment using ArgoCD ApplicationSets.

## Overview

The unified bootstrap chart includes all components from both the original bootstrap and bootstrap-rag deployments:

### Bootstrap Components
- RHOAI Operator & Installation & Configuration
- Shared MinIO, Database, and Milvus services
- LLM serving infrastructure
- PgVector deployment
- Gradio UI
- Image detection services
- User project management
- Application deployment

### Bootstrap RAG Components
- RAG-specific MinIO & Database configurations
- RAG LLM serving
- Gradio RAG interface
- PgVector RAG deployment
- Data Science RAG pipelines
- RAG-specific configurations
- Model server setup
- Workbench images

## Installation

```bash
# Install the unified chart
helm install unified-bootstrap ./unified-bootstrap-chart

# Install with custom values
helm install unified-bootstrap ./unified-bootstrap-chart -f custom-values.yaml

# Upgrade
helm upgrade unified-bootstrap ./unified-bootstrap-chart
```

## Configuration

### Basic Configuration

The chart supports independent control of both bootstrap and bootstrap-rag components:

```yaml
# Enable/disable entire sections
bootstrap:
  enabled: true    # Controls all bootstrap components
bootstrapRag:
  enabled: true    # Controls all bootstrap-rag components

# Global settings
global:
  repoURL: https://github.com/ritzshah/llm-rag-deployment.git
  targetRevision: update-token-v1
  cluster: in-cluster
```

### Component Control

Individual components can be enabled/disabled:

```yaml
bootstrap:
  components:
    - name: gradio
      path: bootstrap/gradio
      enabled: false  # Disable bootstrap gradio

bootstrapRag:
  components:
    - name: workbench
      path: bootstrap-rag/notebook
      enabled: true   # Enable RAG workbench
```

### Gradio Configuration

Shared Gradio configuration applies to both bootstrap and bootstrap-rag deployments:

```yaml
gradio:
  config:
    bearerToken: "your-secure-token"
    inferenceServerUrl: "https://your-server.com/v1/completions"
    modelName: "your-model-name"
    dbConnectionString: "postgresql://user:pass@db:5432/db"
    dbCollectionName: "your_collection"
    maxNewTokens: "1024"
```

## Deployment Strategies

### 1. Full Deployment
Deploy everything (default):
```yaml
bootstrap:
  enabled: true
bootstrapRag:
  enabled: true
```

### 2. Bootstrap Only
Deploy core infrastructure only:
```yaml
bootstrap:
  enabled: true
bootstrapRag:
  enabled: false
```

### 3. RAG Only
Deploy RAG components only (requires bootstrap infrastructure):
```yaml
bootstrap:
  enabled: false
bootstrapRag:
  enabled: true
```

### 4. Custom Deployment
Fine-tune individual components:
```yaml
bootstrap:
  enabled: true
  components:
    - name: gradio
      enabled: false  # Use RAG gradio instead

bootstrapRag:
  enabled: true
  components:
    - name: workbench
      enabled: true   # Enable development workbench
```

## ApplicationSets

The chart creates two ArgoCD ApplicationSets:

1. **bootstrap**: Manages core infrastructure components
2. **bootstrap-rag**: Manages RAG-specific components

Both ApplicationSets share the same sync policy but can be independently controlled.

## Dependencies

- OpenShift cluster with ArgoCD installed
- Sufficient resources for AI/ML workloads
- Access to required container registries

## Upgrading from Separate Charts

If you previously used the separate bootstrap-chart and bootstrap-rag-chart:

1. **Uninstall old charts:**
   ```bash
   helm uninstall bootstrap-deployment
   helm uninstall bootstrap-rag-deployment
   ```

2. **Install unified chart:**
   ```bash
   helm install unified-bootstrap ./unified-bootstrap-chart
   ```

3. **Migrate custom values:**
   - Move `gradio.*` values to the unified structure
   - Combine component lists under `bootstrap.components` and `bootstrapRag.components`

## Troubleshooting

### Check ApplicationSets
```bash
kubectl get applicationsets -n openshift-gitops
kubectl describe applicationset bootstrap -n openshift-gitops
kubectl describe applicationset bootstrap-rag -n openshift-gitops
```

### Check Applications
```bash
kubectl get applications -n openshift-gitops
kubectl get applications -l component=bootstrap -n openshift-gitops
kubectl get applications -l component=bootstrap-rag -n openshift-gitops
```

### View Configuration
```bash
helm get values unified-bootstrap
helm template ./unified-bootstrap-chart --debug
```

## Examples

See the `examples/` directory for common configuration patterns:
- Development environment setup
- Production deployment
- Custom component selection
- Advanced Gradio configuration