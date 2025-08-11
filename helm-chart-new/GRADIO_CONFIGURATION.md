# Gradio Configuration Guide

This document explains how to configure the Gradio UI component through the Helm chart values.

## Overview

The Gradio deployment has been templated to allow easy configuration of key parameters through the `values.yaml` file. All hardcoded values have been extracted and made configurable.

## Key Configuration Parameters

### Required Parameters (mentioned in the request)

| Parameter | Description | Default Value | Location in values.yaml |
|-----------|-------------|---------------|------------------------|
| `BEARER_TOKEN` | Authentication token for the inference server | `65fc80b0d55be557b1365687ddb771d6` | `gradio.config.bearerToken` |
| `INFERENCE_SERVER_URL` | URL of the inference server | `https://granite-8b-code-instruct-maas-apicast...` | `gradio.config.inferenceServerUrl` |
| `MODEL_NAME` | Name of the LLM model to use | `granite-8b-code-instruct-128k` | `gradio.config.modelName` |
| `DB_CONNECTION_STRING` | PostgreSQL database connection string | `postgresql+psycopg://vectordb:vectordb@...` | `gradio.config.dbConnectionString` |
| `DB_COLLECTION_NAME` | Vector database collection name | `documents_test` | `gradio.config.dbCollectionName` |
| `MAX_NEW_TOKENS` | Maximum number of tokens to generate | `512` | `gradio.config.maxNewTokens` |

### Additional Configurable Parameters

| Parameter | Description | Default Value | Location |
|-----------|-------------|---------------|----------|
| `APP_TITLE` | Application title | `Talk with your documentation` | `gradio.config.appTitle` |
| `TOP_K` | Top-K sampling parameter | `10` | `gradio.modelParams.topK` |
| `TOP_P` | Top-P sampling parameter | `0.95` | `gradio.modelParams.topP` |
| `TYPICAL_P` | Typical-P sampling parameter | `0.95` | `gradio.modelParams.typicalP` |
| `TEMPERATURE` | Model temperature | `0.01` | `gradio.modelParams.temperature` |
| `REPETITION_PENALTY` | Repetition penalty | `1.03` | `gradio.modelParams.repetitionPenalty` |
| `ENABLE_DEBUG_UI` | Enable debug interface | `False` | `gradio.debug.enableDebugUI` |

## Configuration Examples

### Basic Configuration

```yaml
# values.yaml
gradio:
  config:
    bearerToken: "your-secure-token-here"
    inferenceServerUrl: "https://your-inference-server.example.com/v1/completions"
    modelName: "your-model-name"
    dbConnectionString: "postgresql+psycopg://user:pass@db-host:5432/dbname"
    dbCollectionName: "your_collection"
    maxNewTokens: "1024"
```

### Advanced Model Parameters

```yaml
# values.yaml
gradio:
  modelParams:
    topK: "50"
    topP: "0.9"
    temperature: "0.7"
    repetitionPenalty: "1.1"
```

### Development Configuration

```yaml
# values-dev.yaml
gradio:
  debug:
    enableDebugUI: "True"
  config:
    appTitle: "DEV - Talk with your documentation"
    maxNewTokens: "256"  # Lower for faster dev cycles
```

### Production Configuration

```yaml
# values-prod.yaml
gradio:
  replicas: 3  # Scale for production
  resources:
    limits:
      cpu: "4"
      memory: 4Gi
    requests:
      cpu: "2"
      memory: 2Gi
  config:
    bearerToken: "{{ .Values.secrets.gradio.bearerToken }}"  # Use secret reference
```

## Deployment Commands

### Install with custom values
```bash
# Using custom values file
helm install gradio-deployment ./bootstrap-chart -f custom-values.yaml

# Override specific values
helm install gradio-deployment ./bootstrap-chart \
  --set gradio.config.bearerToken="new-token" \
  --set gradio.config.modelName="new-model"
```

### Upgrade configuration
```bash
helm upgrade gradio-deployment ./bootstrap-chart -f updated-values.yaml
```

## Security Considerations

1. **Bearer Token**: Always change the default bearer token in production
2. **Database Credentials**: Consider using Kubernetes secrets for sensitive database connection strings
3. **Debug Mode**: Ensure debug UI is disabled in production environments

## Chart Locations

- **Bootstrap Chart**: `helm-chart-new/bootstrap-chart/` (main gradio deployment)
- **Bootstrap RAG Chart**: `helm-chart-new/bootstrap-rag-chart/` (RAG-specific gradio deployment)

Both charts use the same configuration structure, allowing consistent management across deployments.

## Troubleshooting

### Check current configuration
```bash
helm get values gradio-deployment
```

### Validate configuration
```bash
helm template ./bootstrap-chart -f your-values.yaml | grep -A 20 "env:"
```

### Common Issues
- **Connection Errors**: Check `inferenceServerUrl` and `dbConnectionString`
- **Authentication Failures**: Verify `bearerToken` is correct
- **Performance Issues**: Adjust `maxNewTokens` and resource limits