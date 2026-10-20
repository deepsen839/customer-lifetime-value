# Streamlit Community Cloud deployment (managed PostgreSQL)

## Deployment scope and current limitations

- Entrypoint: `app.py` (project root).
- Python: choose **3.11** in Streamlit Community Cloud Advanced settings to match the Docker image and reduce dependency drift.
- Database credentials are read from `DATABASE_URL` in Streamlit Secrets, with the environment-variable path retained for Docker/local use.
- Do not commit `.streamlit/secrets.toml`, `.env`, downloaded datasets, or private database credentials.
- Community Cloud does not run `docker-compose.yml`; it runs the Python app. FastAPI and the local PostgreSQL container are not deployed by this setup.
- The app downloads IBM Telco on first use. Community Cloud's local filesystem is ephemeral, so downloaded datasets and locally generated artifacts may disappear after a restart/rebuild.
- The model pipeline outputs are gitignored under `artifacts/`; unless you run training and deliberately include small, non-sensitive output files, the deployed app will show “Run training first” messages for survival/uplift/CLV tabs. The current app is a dashboard, not an automated cloud training job.
- The PostgreSQL tab creates the schema and reads tables, but it does not load datasets into the database by itself. To populate tables, run `python -m src.train_real` in a trusted environment with the managed `DATABASE_URL` configured. This pipeline can be long-running and downloads multiple datasets.

## 1. Create a managed PostgreSQL database

Use a managed provider such as Neon or Supabase. Create a project/database, copy its PostgreSQL connection string, and ensure external connections are enabled. Prefer a pooled or standard connection string supported by the provider; keep SSL enabled (`sslmode=require`). Do not paste credentials into GitHub, issues, screenshots, or chat.

## 2. Prepare and test the project locally

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Test the database connection before deployment. Set `DATABASE_URL` in your shell, using the provider's real connection string:

```bash
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require'
python - <<'PY'
from src.db import get_engine, init_db
engine = get_engine()
if engine is None:
    raise SystemExit('DATABASE_URL was not found')
with engine.connect() as conn:
    print(conn.exec_driver_sql('select version()').scalar())
init_db(engine)
print('Database connectivity and schema initialization passed')
PY
```

If the provider gives a `postgres://` or `postgresql://` URL, `src/db.py` normalizes it to the installed psycopg v3 driver. If the password contains URL-reserved characters, use the provider's URL-encoded connection string or the discrete `[database]` secret fields shown in `.streamlit/secrets.toml.example`.

## 3. Create GitHub repository

1. Sign in to GitHub and create an **empty** repository, e.g. `customer-lifetime-value-churn-intelligence`.
2. Do not initialize it with a README/license if you already have project files locally.
3. Confirm `.gitignore` excludes `.env`, `.streamlit/secrets.toml`, `data/*`, `artifacts/*`, Python caches, and local virtual environments.
4. From the project root, inspect `git status` and make sure no secrets, private data, or oversized files are staged.
5. Commit and push:

```bash
git init
git branch -M main
git add .
git status --short
# Review staged files carefully before committing.
git commit -m "Prepare CLV and churn intelligence dashboard for deployment"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/customer-lifetime-value-churn-intelligence.git
git push -u origin main
```

If this folder is already a Git repository, do not run `git init` or add a second remote; inspect `git status` and `git remote -v` first.

## 4. Deploy on Streamlit Community Cloud

1. Open <https://share.streamlit.io/> and sign in with GitHub.
2. Select **Create app**. Choose the repository, branch `main`, and entrypoint `app.py`.
3. Open **Advanced settings**, select Python **3.11**, and paste the secret below after replacing all placeholders with the real managed PostgreSQL values.
4. Click **Deploy** and watch the deployment logs.

```toml
DATABASE_URL = "postgresql+psycopg://DB_USER:URL_ENCODED_PASSWORD@DB_HOST:5432/DB_NAME?sslmode=require"
```

Use the exact connection URL from your provider, and URL-encode reserved characters in the username/password if necessary. Do not store the real URL in a committed file.

## 5. Verify after deployment

- The app opens and the Overview & EDA tab loads IBM Telco (first load may take a little while).
- Churn + SHAP shows model evaluation only if the corresponding generated artifact files are present; otherwise its training message is expected.
- PostgreSQL tab reports no database error and lists the schema tables. Tables can exist but remain empty until the training pipeline has loaded data.
- Check Community Cloud logs for missing packages, network/TLS errors, or database authentication errors.
- Re-test the database connection locally if PostgreSQL fails; check provider allow-lists, database name, user permissions, SSL mode, and connection string.

## 6. Populate the managed database (optional, run from a trusted machine)

After testing the managed connection, install project requirements locally and run:

```bash
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require'
python -m src.train_real
```

This is a full data download/training workflow, not a quick deployment health check. It can take time and uses the Hillstrom and Online Retail II source URLs; the Hillstrom source has previously failed TLS in the user's environment. Do not claim that the full pipeline or all datasets are verified unless it completes successfully.

## Rollback / recovery

- A failed build: open the app's deployment logs and fix the first dependency/import error; commit and push the fix.
- A database failure: edit App settings > Secrets and verify the provider URL; restart/redeploy the app.
- Never temporarily disable TLS certificate verification to make a dataset download work.
