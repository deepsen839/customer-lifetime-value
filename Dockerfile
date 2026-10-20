FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import lifelines; print('lifelines dependency OK:', lifelines.__version__)" && \
    python -c "import numpy, pyarrow; assert numpy.__version__.split('.')[0] == '1', f'Expected NumPy 1.x, got {numpy.__version__}'; assert pyarrow.__version__ == '18.1.0', f'Expected PyArrow 18.1.0, got {pyarrow.__version__}'; print(f'NumPy/PyArrow compatibility OK: NumPy {numpy.__version__}, PyArrow {pyarrow.__version__}')"
COPY . .
RUN mkdir -p /app/data /app/artifacts && useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501 8000
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]
