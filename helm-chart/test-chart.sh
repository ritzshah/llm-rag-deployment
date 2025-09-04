#!/bin/bash

# Helm Chart Validation and Testing Script
# This script validates the Helm chart templates and performs basic testing

set -e

CHART_DIR="$(dirname "$0")"
CHART_NAME="gradio-llm-rag"

echo "🚀 Testing Helm Chart: $CHART_NAME"
echo "📁 Chart Directory: $CHART_DIR"

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check if Helm is installed
if ! command -v helm &> /dev/null; then
    print_error "Helm is not installed. Please install Helm first."
    exit 1
fi

print_success "Helm is installed: $(helm version --short)"

# Lint the chart
print_status "Linting Helm chart..."
if helm lint "$CHART_DIR"; then
    print_success "Chart linting passed"
else
    print_error "Chart linting failed"
    exit 1
fi

# Template the chart with default values
print_status "Templating chart with default values..."
if helm template test-release "$CHART_DIR" > /tmp/helm-template-default.yaml; then
    print_success "Default templating successful"
    echo "Output saved to: /tmp/helm-template-default.yaml"
else
    print_error "Default templating failed"
    exit 1
fi

# Template the chart with development values
if [ -f "$CHART_DIR/values-dev.yaml" ]; then
    print_status "Templating chart with development values..."
    if helm template test-release "$CHART_DIR" -f "$CHART_DIR/values-dev.yaml" > /tmp/helm-template-dev.yaml; then
        print_success "Development templating successful"
        echo "Output saved to: /tmp/helm-template-dev.yaml"
    else
        print_error "Development templating failed"
        exit 1
    fi
fi

# Template the chart with production values
if [ -f "$CHART_DIR/values-prod.yaml" ]; then
    print_status "Templating chart with production values..."
    if helm template test-release "$CHART_DIR" -f "$CHART_DIR/values-prod.yaml" > /tmp/helm-template-prod.yaml; then
        print_success "Production templating successful"
        echo "Output saved to: /tmp/helm-template-prod.yaml"
    else
        print_error "Production templating failed"
        exit 1
    fi
fi

# Validate Kubernetes manifests (if kubectl is available)
if command -v kubectl &> /dev/null; then
    print_status "Validating generated Kubernetes manifests..."
    
    # Check default template
    if kubectl apply --dry-run=client -f /tmp/helm-template-default.yaml > /dev/null 2>&1; then
        print_success "Default manifest validation passed"
    else
        print_error "Default manifest validation failed"
        exit 1
    fi
    
    # Check dev template if it exists
    if [ -f /tmp/helm-template-dev.yaml ]; then
        if kubectl apply --dry-run=client -f /tmp/helm-template-dev.yaml > /dev/null 2>&1; then
            print_success "Development manifest validation passed"
        else
            print_error "Development manifest validation failed"
            exit 1
        fi
    fi
    
    # Check prod template if it exists
    if [ -f /tmp/helm-template-prod.yaml ]; then
        if kubectl apply --dry-run=client -f /tmp/helm-template-prod.yaml > /dev/null 2>&1; then
            print_success "Production manifest validation passed"
        else
            print_error "Production manifest validation failed"
            exit 1
        fi
    fi
else
    print_status "kubectl not found, skipping Kubernetes manifest validation"
fi

# Test with custom values
print_status "Testing with custom values..."
if helm template test-release "$CHART_DIR" \
    --set replicaCount=2 \
    --set image.tag=test \
    --set app.title="Custom Test App" > /tmp/helm-template-custom.yaml; then
    print_success "Custom values templating successful"
    echo "Output saved to: /tmp/helm-template-custom.yaml"
else
    print_error "Custom values templating failed"
    exit 1
fi

# Show what would be installed
print_status "Preview of resources that would be created:"
helm template test-release "$CHART_DIR" | grep -E "^kind:|^  name:" | paste - - | column -t

print_success "🎉 All tests passed!"
print_status "Installation commands:"
echo "  Development: helm install my-app $CHART_DIR -f $CHART_DIR/values-dev.yaml"
echo "  Production:  helm install my-app $CHART_DIR -f $CHART_DIR/values-prod.yaml"
echo "  Default:     helm install my-app $CHART_DIR" 