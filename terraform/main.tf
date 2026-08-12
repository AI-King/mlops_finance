# This starter creates a namespace only. Connect it to your cloud provider's
# Kubernetes cluster after authentication and provider configuration.
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
  }
}

variable "namespace" {
  description = "Kubernetes namespace for the ML service."
  type        = string
  default     = "mlops-finance"
}

resource "kubernetes_namespace" "mlops" {
  metadata {
    # Namespaces isolate the application from unrelated workloads.
    name = var.namespace
  }
}
