FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy script and requirements
COPY ./src .
COPY requirements.txt .
COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

# Command to run the script
CMD ["python", "main.py"]