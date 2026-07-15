# Multi-Hop RAG Production Deployment Guide

This guide details step-by-step instructions for deploying both the backend FastAPI service and the frontend React application to production.

---

## 1. Prerequisites & Preparation

Before starting, make sure you have the following API keys and accounts:
- [ ] **Groq Account**: To get a `GROQ_API_KEY` for query decomposition.
- [ ] **Google AI Studio**: To get a `GEMINI_API_KEY` for answer synthesis.
- [ ] **Qdrant Cloud**: A free-tier cluster to host the vector database.
- [ ] **Render** (or Fly.io) Account: For hosting the containerized backend.
- [ ] **Vercel** Account: For hosting the static frontend.

---

## 2. Setting Up Qdrant Cloud (Vector Database)

1. Sign up/log in at [Qdrant Cloud Console](https://cloud.qdrant.io/).
2. Create a new cluster (the free tier includes 1 cluster with 1GB RAM / 0.5 vCPU, which is perfect for this project).
3. Once the cluster is running:
   - Copy the **Endpoint URL** (looks like `https://xxxxxx.gcp.qdrant.io:6333` or similar). This is your `QDRANT_URL`.
   - Generate an **API Key** under the cluster details. This is your `QDRANT_API_KEY`.
4. *Keep these values handy for the next steps.*

---

## 3. Backend Container Deployment (Render)

Render automatically builds and runs the container from the [Dockerfile](file:///d:/Projects/multihop-rag/Dockerfile) in the repository root.

### Step-by-Step Render Setup:
1. Log in to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New** -> **Web Service**.
3. Connect your GitHub/GitLab repository where the code is pushed.
4. Configure the service settings:
   - **Name**: `multihop-rag-backend` (or custom name)
   - **Region**: Select a region close to your user base.
   - **Branch**: `main` (or your active feature branch)
   - **Runtime**: `Docker`
   - **Instance Type**: **Starter** (512MB or 1GB RAM) is recommended. 
     *(Note: Hugging Face models running on CPU require standard system memory. The Free tier has 512MB RAM, which might hit memory limits when initializing PyTorch. If using Free tier, ensure `MOCK_MODELS=true` is set if you run out of memory, or select the Starter tier).*
5. Add the following **Environment Variables** under the "Environment" tab:
   - `GROQ_API_KEY`: *(Your Groq API key)*
   - `GEMINI_API_KEY`: *(Your Google Gemini API key)*
   - `QDRANT_URL`: *(Your Qdrant Cloud URL)*
   - `QDRANT_API_KEY`: *(Your Qdrant Cloud API Key)*
   - `ALLOWED_ORIGINS`: `https://your-frontend-domain.vercel.app` (or `*` temporarily to allow all domains)
6. Click **Deploy Web Service**.

### Post-Deploy Smoke Test (Verification):
Once Render completes building and running the container, verify it is healthy:
1. Copy the deployed service URL (e.g., `https://multihop-rag-backend.onrender.com`).
2. Make a GET request to `/health` in your browser or terminal:
   ```bash
   curl https://multihop-rag-backend.onrender.com/health
   ```
3. A successful deploy will return:
   ```json
   {
     "status": "healthy",
     "qdrant_connectivity": "connected"
   }
   ```

---

## 4. Frontend Deployment (Vercel)

Vercel will build and serve the React Single Page Application (SPA).

### Step-by-Step Vercel Setup:
1. Log in to the [Vercel Dashboard](https://vercel.com/).
2. Click **Add New** -> **Project**.
3. Import your Git repository.
4. Configure the project:
   - **Framework Preset**: `Vite` (automatically detected)
   - **Root Directory**: `frontend` (Click **Edit** next to project root and select the `frontend` folder)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Expand the **Environment Variables** section and add:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: `https://multihop-rag-backend.onrender.com` *(The URL of your deployed Render backend)*
6. Click **Deploy**.
7. Vercel will build and assign a domain (e.g., `https://multihop-rag-frontend.vercel.app`).
8. *(Optional)* Go back to your Render backend configuration and update `ALLOWED_ORIGINS` to match your Vercel deployment URL for tighter security.

---

## 5. Local Docker Testing (Pre-Deployment Validation)

To build and run the production image locally:

1. **Build the image**:
   ```bash
   docker build -t multihop-rag-backend .
   ```
2. **Run the container**:
   Provide the required variables via `-e`.
   ```bash
   docker run -d -p 8000:8000 \
     -e GROQ_API_KEY="your-groq-key" \
     -e GEMINI_API_KEY="your-gemini-key" \
     -e QDRANT_URL="https://your-qdrant-cloud-url.qdrant.io:6333" \
     -e QDRANT_API_KEY="your-qdrant-api-key" \
     -e ALLOWED_ORIGINS="*" \
     -e PORT=8000 \
     --name multihop-backend-test \
     multihop-rag-backend
   ```
3. Test locally at [http://localhost:8000/health](http://localhost:8000/health).
