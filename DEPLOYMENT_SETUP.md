# Deployment Setup Instructions

## IMPORTANT: Docker Build Fix Applied

**Issue Found**: The original requirements.txt included `playwright` and `mcp` which require additional system dependencies not available in Python slim containers.

**Fix Applied**: Created `requirements-docker.txt` without these problematic dependencies and updated Dockerfile to use it.

## Required GitHub Secrets

To use the deployment workflow, you need to configure these secrets in your GitHub repository:

### Repository Settings > Secrets and variables > Actions

1. **DEPLOY_HOST** - The hostname or IP address of your production server
2. **DEPLOY_USER** - The SSH username for your production server
3. **DEPLOY_SSH_KEY** - The private SSH key for authenticating to your production server

### Example Values:
```
DEPLOY_HOST: your-server.example.com
DEPLOY_USER: ubuntu
DEPLOY_SSH_KEY: -----BEGIN OPENSSH PRIVATE KEY-----
[your private key content]
-----END OPENSSH PRIVATE KEY-----
```

## Server Requirements

Your production server needs:
- Docker installed and running
- SSH access configured
- User has permission to run Docker commands
- Port 8501 available for the Streamlit application

## Manual Testing (REQUIRED BEFORE PUSH)

**CRITICAL**: Test locally first to ensure Docker build works:

```bash
# Build the image (should complete without errors)
docker build -t loop-test .

# Run the container
docker run -p 8501:8501 loop-test

# You should see output like:
# Docker startup: Creating sample data...
# Using datasets/klines_2h_2020_2025.csv as sample data source
# Created sample data at /tmp/historical_data.parquet
# Data shape: (23353, 19)
# Docker startup: Sample data created successfully
# Starting Streamlit app...
```

Then visit http://localhost:8501 to see the Streamlit application.

## Files Modified/Created

- `Dockerfile` - Fixed to use requirements-docker.txt and added system dependencies
- `requirements-docker.txt` - Clean dependencies without playwright/mcp
- `docker_startup.py` - Creates sample data from CSV and launches Streamlit
- `.github/workflows/deploy.yml` - Complete deployment pipeline