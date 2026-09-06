# Deploying to Kubernetes (optional)

This is an optional add-on, not part of the graded core build. `docker compose up` is the
primary, required way to run this project — these manifests exist for anyone who wants to
see the deployment shape on a real cluster, nothing more.

These manifests describe how this system would be deployed to a real cluster. **They
are not applied to any live cluster as part of this submission** — this doc exists so any
engineer can stand it up on their own cluster in a few minutes.

## Prerequisites

- A Kubernetes cluster (GKE, EKS, AKS, or local — `minikube` / `kind` work fine for a demo)
- `kubectl` pointed at that cluster
- A container registry you can push to (Docker Hub, GCR/Artifact Registry, ECR, etc.)
- An ingress controller installed on the cluster if you want external access
  (e.g. `kubectl apply -f https://.../ingress-nginx/deploy.yaml`) — skip this and use
  `kubectl port-forward` if you just want to poke at it locally

## 1. Build and push the image

```bash
docker build -t <your-registry>/investment-committee-api:latest .
docker push <your-registry>/investment-committee-api:latest
```

Then edit `k8s/api.yaml` and replace `REPLACE_WITH_YOUR_REGISTRY/investment-committee-api:latest`
with the image you just pushed.

## 2. Create the secret

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
```

Fill in `LLM_API_KEY` (and `GRAFANA_ADMIN_PASSWORD`) in `k8s/secret.yaml`. **Do not commit
this file** — it's already excluded via `.gitignore`. Then edit `k8s/kustomization.yaml` to
point at `secret.yaml` instead of `secret.example.yaml`.

## 3. Apply everything

```bash
kubectl apply -k k8s/
```

This creates the `investment-committee` namespace and everything inside it. If you'd rather
apply pieces individually (useful for understanding dependency order):

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/dependencies.yaml      # Mongo + Redis first
kubectl apply -f k8s/observability.yaml     # Prometheus + Grafana + MLflow
kubectl apply -f k8s/api.yaml               # API last, it depends on the above
kubectl apply -f k8s/ingress.yaml           # optional
```

## 4. Verify

```bash
kubectl get pods -n investment-committee
kubectl logs -f deployment/committee-api -n investment-committee
```

Without an ingress, reach things via port-forward:

```bash
kubectl port-forward svc/committee-api 8000:80 -n investment-committee   # API
kubectl port-forward svc/grafana 3000:3000 -n investment-committee       # Grafana
kubectl port-forward svc/mlflow 5000:5000 -n investment-committee        # MLflow
```

## Switching the AI foundation model

This is the actual "any engineer can plug in a different provider" requirement — it's a
config change, not a code change:

```bash
# edit LLM_PROVIDER / LLM_MODEL in k8s/configmap.yaml (and LLM_API_KEY in secret.yaml
# if you're also switching providers, since the key format differs)
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml       # only if the key changed
kubectl rollout restart deployment/committee-api -n investment-committee
```

`committee.llm.client` reads `LLM_PROVIDER` at startup and dispatches to the matching
backend (`anthropic`, `openai`, or `litellm` for anything else) — no agent or orchestrator
code references a provider SDK directly, so this is the only place that needs to change.

**Vertex AI (GCP service-account) specifically** needs one more step beyond the config-only
swap above: `LLM_VERTEX_PROJECT`/`LLM_VERTEX_LOCATION` are already in `configmap.yaml`, but
Vertex auth needs `GOOGLE_APPLICATION_CREDENTIALS` to point at an actual service-account
JSON *file* inside the container, not just an env var — a plain `Secret` env entry (like
`LLM_API_KEY`) can't carry a file. Mount the key as a Secret volume instead:

```bash
kubectl create secret generic gcp-sa-key -n investment-committee --from-file=key.json=./gemini_key.json
```

then add to `api.yaml`'s Deployment spec:

```yaml
env:
  - name: GOOGLE_APPLICATION_CREDENTIALS
    value: /var/secrets/google/key.json
volumeMounts:
  - name: gcp-sa-key
    mountPath: /var/secrets/google
    readOnly: true
volumes:
  - name: gcp-sa-key
    secret:
      secretName: gcp-sa-key
```

Not wired into `api.yaml` by default since Anthropic (no file-based auth needed) is this
project's documented default provider — noted here as the concrete next step for anyone
actually deploying the Vertex AI path this build was developed against, rather than left
as a silent gap.

## What's intentionally left out

Documented here rather than silently missing, since these are real gaps for a production
deployment and it matters that they're a choice, not an oversight:

- **No TLS / cert-manager** — the Ingress serves plain HTTP. Add `cert-manager` and a
  `ClusterIssuer` plus a `tls:` block on the Ingress for a real deployment.
- **No NetworkPolicies** — all pods in the namespace can currently reach each other and,
  depending on cluster config, the outside world. Fine for a demo namespace, not for
  multi-tenant clusters.
- **No PodDisruptionBudget** — a node drain could take down both API replicas
  simultaneously. Add a PDB with `minAvailable: 1` before relying on this for uptime.
- **Self-hosted Mongo/Redis instead of managed services** — see the comment at the top of
  `dependencies.yaml`. Fine for demonstrating the deployment shape; swap for
  MongoDB Atlas / Memorystore before anything real depends on this.
- **CPU-based HPA on an I/O-bound workload** — see the comment in `api.yaml`. A latency- or
  concurrency-based signal (via the Prometheus Adapter or KEDA) would scale more accurately.
- **No image vulnerability scanning or admission policy** — add one before pushing this
  past a personal cluster.
