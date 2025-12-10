FROM python:3.12-slim-bookworm

WORKDIR /api

COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git git-lfs && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install 'litellm[proxy]' -U && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 4000

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
