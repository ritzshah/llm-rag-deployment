{{/*
Expand the name of the chart.
*/}}
{{- define "unified-bootstrap-chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "unified-bootstrap-chart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "unified-bootstrap-chart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "unified-bootstrap-chart.labels" -}}
helm.sh/chart: {{ include "unified-bootstrap-chart.chart" . }}
{{ include "unified-bootstrap-chart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "unified-bootstrap-chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "unified-bootstrap-chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Legacy bootstrap-chart helpers for compatibility
*/}}
{{- define "bootstrap-chart.name" -}}
{{- include "unified-bootstrap-chart.name" . }}
{{- end }}

{{- define "bootstrap-chart.fullname" -}}
{{- include "unified-bootstrap-chart.fullname" . }}
{{- end }}

{{- define "bootstrap-chart.chart" -}}
{{- include "unified-bootstrap-chart.chart" . }}
{{- end }}

{{- define "bootstrap-chart.labels" -}}
{{- include "unified-bootstrap-chart.labels" . }}
{{- end }}

{{- define "bootstrap-chart.selectorLabels" -}}
{{- include "unified-bootstrap-chart.selectorLabels" . }}
{{- end }}

{{/*
Legacy bootstrap-rag-chart helpers for compatibility
*/}}
{{- define "bootstrap-rag-chart.name" -}}
{{- include "unified-bootstrap-chart.name" . }}
{{- end }}

{{- define "bootstrap-rag-chart.fullname" -}}
{{- include "unified-bootstrap-chart.fullname" . }}
{{- end }}

{{- define "bootstrap-rag-chart.chart" -}}
{{- include "unified-bootstrap-chart.chart" . }}
{{- end }}

{{- define "bootstrap-rag-chart.labels" -}}
{{- include "unified-bootstrap-chart.labels" . }}
{{- end }}

{{- define "bootstrap-rag-chart.selectorLabels" -}}
{{- include "unified-bootstrap-chart.selectorLabels" . }}
{{- end }}