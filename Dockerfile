FROM python:3.12-slim

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY server.py .

# Port exposé par Railway
ENV PORT=8000
ENV MCP_TRANSPORT=streamable-http

EXPOSE 8000

CMD ["python", "server.py"]
