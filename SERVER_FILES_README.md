# GCP Server Folder Structure

This README documents the files and folders currently present on the GCP Server under the `aiautono247` user's home directory.

## 1. `~/uyf-crm`
This folder is responsible for running the CRM application (both Backend and Frontend).
- **`crm-backend/`**
  - `backend.tar` (267 MB) - The compiled Docker image/files for the CRM Spring Boot backend.
- **`crm-frontend/`**
  - `dist/` - The compiled static build files (HTML, CSS, JS) for the CRM React frontend.
- **`deploy/`**
  - Contains the `docker-compose.yml` and `.env` files used to start the CRM backend and frontend containers.

## 2. `~/uyf-finance`
This folder is responsible for running the Finance application.
- **`.env`**
  - Contains the injected environment variables (like `CRM_WEBHOOK_SECRET`) for the Finance app.
- **`docker-compose.yml`**
  - The configuration file used to run the `uyf-finance` Docker container.

---
*Note: These files are automatically updated by GitHub Actions during the CI/CD deployment process.*
