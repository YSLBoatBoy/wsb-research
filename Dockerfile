FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WSB_DATA_DIR=/data \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=America/Chicago \
    PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py ./

VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=4)" || exit 1

CMD ["python", "server.py"]
