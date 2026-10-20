# Docker quick start

Requires Docker Engine/Desktop and Docker Compose v2.

1. Copy `.env.example` to `.env`.
2. Change `POSTGRES_PASSWORD` to a strong password. Never commit `.env`.
3. From the project root run:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f app
```

Open http://localhost:8501.

## Generate demo data and train
```bash
docker compose exec app python -m src.generate_data --rows 3000 --output /app/data/demo_customers.csv
docker compose exec app python -m src.train
```
Refresh the browser. The default data is synthetic and its metrics are not real-world evidence.

## Load data into PostgreSQL
```bash
docker compose exec app python -m src.load_data /app/data/demo_customers.csv
```
Warning: the current loader replaces the `customers` table during load. Run only when replacement is intended.

## Health checks and tests
```bash
docker compose ps
docker compose exec app python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').read().decode())"
docker compose exec db sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
docker compose exec app pytest -q
```

## Stop
```bash
docker compose down
```
To also delete persisted volumes (irreversible data loss):
```bash
docker compose down -v
```

For IBM Telco column mapping, see `docs/DATA_DICTIONARY.md`.
