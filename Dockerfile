FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --system --create-home agent \
    && install -d -o agent -g agent /var/lib/flight-agent
COPY flight_agent flight_agent
COPY tests tests
USER agent
CMD ["python", "-m", "flight_agent.cli"]
