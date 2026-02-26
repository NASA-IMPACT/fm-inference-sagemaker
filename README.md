# Unified Docker Deployment Script

A single script that handles building, pushing to ECR, and cleaning up old images.

## Quick Start

```bash
# Set required variables
export ECR_URL="123456789.dkr.ecr.us-west-2.amazonaws.com"
export INGRESS_HOST="your-ingress-host.com"
export DATABASE_URL="postgresql://..."

# Run deployment
./deploy_robust.sh
```

## What It Does

1. **Builds** all Docker images using multistage builds and layer caching
2. **Pushes** to ECR with dual tagging (digest + latest)
3. **Cleans up** old ECR images (keeps 5 most recent)
4. **Prunes** local disk usage (dangling layers and old build cache)
5. **Deploys** to Kubernetes

## Configuration

Control behavior with environment variables:

```bash
# Keep 10 images instead of 5
MAX_IMAGES_TO_KEEP=10 ./deploy_robust.sh

# Preview cleanup without deleting
DRY_RUN_CLEANUP=true ./deploy_robust.sh

# Automatically remove locally tagged images after successful push
CLEANUP_AFTER_PUSH=true ./deploy_robust.sh

# Test build locally without pushing or deploying
SKIP_PUSH=true SKIP_DEPLOY=true ./deploy_robust.sh
```

## Multistage Builds

The script utilizes a multistage build process defined in `Dockerfile.multistage` to ensure minimal image sizes without the reliability risks of post-processing tools.

- **Builder Stage**: Compiles heavy dependencies (GDAL, PyTorch).
- **Runtime Stage**: Copies only final libraries and application code.
- **Typical sizes**: 1.5GB - 2.0GB (standard for ML/Geospatial stacks).

## Cleanup Strategy

Automatically removes:
- ✅ Old ECR images (keeps 5 newest per repo)
- ✅ Local dangling images
- ✅ Unused build cache (>24h old)
- ✅ Temporary build images
- ✅ Locally built/tagged images after successful push (if `CLEANUP_AFTER_PUSH=true`)

Preserves:
- ✅ 5 most recent versions in ECR (configurable)
- ✅ Currently deployed images
- ✅ Build cache for frequently modified files

## Build Performance

With caching enabled:
- **First build**: 30-40 minutes (compiling GDAL/PyTorch)
- **Subsequent builds**: 1-5 minutes (utilizing ECR cache layers)
- **Minor changes**: < 1 minute (source code only)

## Repositories Managed

- inference
- inference_pipelines/base
- inference_pipelines/floods
- inference_pipelines/burn_scars
- inference_pipelines/crop_classification
- inference_pipelines/surya_rollout
- tile_server/tiler

## Requirements

- Docker
- AWS CLI configured
- kubectl configured
- gettext (for `envsubst`)

## CI/CD Integration Example

```yaml
# GitHub Actions example
- name: Deploy
  env:
    ECR_URL: ${{ secrets.ECR_URL }}
    INGRESS_HOST: ${{ secrets.INGRESS_HOST }}
    CLEANUP_AFTER_PUSH: true
  run: ./deploy_robust.sh
```