# Gradio LLM RAG Helm Chart

This Helm chart deploys a Gradio-based LLM RAG (Retrieval-Augmented Generation) application with PostgreSQL vector database integration.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- OpenShift 4.x (if using OpenShift Routes)

## Installation

### Basic Installation

```bash
helm install my-gradio-app ./helm-chart
```

### Installation with Custom Values

```bash
helm install my-gradio-app ./helm-chart -f custom-values.yaml
```

### Installation with Inline Values

```bash
helm install my-gradio-app ./helm-chart \
  --set image.tag=v1.0.0 \
  --set replicaCount=2 \
  --set llm.modelName=my-custom-model
```

## Configuration

The following table lists the configurable parameters and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `quay.io/rshah/llm-rag-deployment-maas` |
| `image.tag` | Container image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `app.title` | Application title | `Talk with your documentation` |
| `app.enableDebugUI` | Enable debug UI | `false` |
| `llm.inferenceServerUrl` | LLM inference server URL | `https://granite-8b-code-instruct-maas-apicast-production.apps.llmaas.llmaas.redhatworkshops.io:443/v1/completions` |
| `llm.modelName` | LLM model name | `granite-8b-code-instruct-128k` |
| `llm.bearerToken` | Bearer token for LLM API | `65fc80b0d55be557b1365687ddb771d6` |
| `database.connectionString` | Database connection string | `postgresql+psycopg://vectordb:vectordb@postgresql-service.ic-shared-llm.svc.cluster.local:5432/vectordb` |
| `database.collectionName` | Database collection name | `documents_test` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `7860` |
| `route.enabled` | Enable OpenShift route | `true` |
| `route.host` | Route hostname (empty for auto-generated) | `""` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.limits.memory` | Memory limit | `2Gi` |
| `resources.requests.cpu` | CPU request | `1` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `namespace.name` | Target namespace | `ic-shared-llm` |
| `namespace.create` | Create namespace if it doesn't exist | `false` |

## Security Considerations

### Secrets Management

For production deployments, consider using Kubernetes secrets for sensitive data:

```yaml
# custom-values.yaml
llm:
  bearerToken: ""  # Leave empty to use from secret

# Create a secret separately
apiVersion: v1
kind: Secret
metadata:
  name: llm-credentials
type: Opaque
data:
  bearerToken: <base64-encoded-token>
```

### Security Context

The chart includes security best practices:
- Runs as non-root user
- Drops all capabilities
- Disables privilege escalation
- Uses RuntimeDefault seccomp profile

## Customization Examples

### Development Environment

```yaml
# dev-values.yaml
replicaCount: 1
app:
  enableDebugUI: true
resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

### Production Environment

```yaml
# prod-values.yaml
replicaCount: 3
image:
  tag: v1.0.0
  pullPolicy: Always
resources:
  requests:
    cpu: 1
    memory: 1Gi
  limits:
    cpu: 2
    memory: 2Gi
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - gradio
        topologyKey: kubernetes.io/hostname
```

## Monitoring and Observability

The application exposes health check endpoints:
- Readiness probe: `GET /`
- Liveness probe: `GET /`
- Startup probe: `GET /`

## Troubleshooting

### Common Issues

1. **Pod not starting**: Check resource limits and node capacity
2. **Database connection issues**: Verify database connection string and network policies
3. **LLM API issues**: Verify bearer token and inference server availability

### Debugging Commands

```bash
# Check pod status
kubectl get pods -n ic-shared-llm

# View pod logs
kubectl logs -n ic-shared-llm deployment/my-gradio-app

# Describe deployment
kubectl describe deployment -n ic-shared-llm my-gradio-app

# Port forward for local testing
kubectl port-forward -n ic-shared-llm service/my-gradio-app-service 7860:7860
```

## Upgrading

```bash
helm upgrade my-gradio-app ./helm-chart -f custom-values.yaml
```

## Uninstalling

```bash
helm uninstall my-gradio-app
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the chart
5. Submit a pull request

## License

This chart is distributed under the same license as the main project. 