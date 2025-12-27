# Docker Setup for PLN ML System

Docker configuration untuk running RabbitMQ dan ML Worker.

## Prerequisites

- Docker installed
- RabbitMQ running (or use Docker)

## Docker Images

### 1. Worker Image (RabbitMQ Consumer)

**Build:**
```bash
docker build -f Dockerfile.worker -t pln-ml-worker .
# atau
docker build -t pln-ml-worker .
```

**Run:**
```bash
# With environment variables
docker run -d \
  --name pln-ml-worker \
  -e RABBITMQ_HOST=your-rabbitmq-host \
  -e RABBITMQ_PORT=5672 \
  -e RABBITMQ_USER=guest \
  -e RABBITMQ_PASS=guest \
  pln-ml-worker

# With .env file
docker run -d \
  --name pln-ml-worker \
  --env-file .env \
  pln-ml-worker
```

### 2. Trainer Image (Training Environment)

**Build:**
```bash
docker build -f Dockerfile.trainer -t pln-ml-trainer .
```

**Run:**
```bash
# Interactive shell untuk running training scripts
docker run -it --rm \
  -v $(pwd)/postpaid/data:/app/postpaid/data \
  -v $(pwd)/postpaid/models:/app/postpaid/models \
  -v $(pwd)/prepaid/data:/app/prepaid/data \
  -v $(pwd)/prepaid/models:/app/prepaid/models \
  pln-ml-trainer bash
```

Inside container, jalankan training:
```bash
# Postpaid training
python postpaid/training/train_anomaly_score.py
python postpaid/training/train_is_anomaly.py
python postpaid/training/train_final_score_rumus1.py
python postpaid/training/train_score_rumus2.py
python postpaid/training/train_final_score_rumus2.py

# Prepaid training
python prepaid/training/train_is_rutin.py
python prepaid/training/train_final_score_rumus1.py
python prepaid/training/train_final_score_rumus2.py
python prepaid/training/train_anomaly_score.py
```

### 3. RabbitMQ (Optional - jika belum ada)

```bash
docker run -d \
  --name pln-rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management
```

Access management UI: http://localhost:15672 (guest/guest)

## Training Workflow

### One-off Training Command

```bash
# Train single model
docker run --rm \
  -v $(pwd)/postpaid/data:/app/postpaid/data \
  -v $(pwd)/postpaid/models:/app/postpaid/models \
  pln-ml-trainer \
  python postpaid/training/train_anomaly_score.py

# Train prepaid model
docker run --rm \
  -v $(pwd)/prepaid/data:/app/prepaid/data \
  -v $(pwd)/prepaid/models:/app/prepaid/models \
  pln-ml-trainer \
  python prepaid/training/train_is_rutin.py
```

### Batch Training Script

Buat script `train-all.sh`:
```bash
#!/bin/bash
docker run --rm \
  -v $(pwd)/postpaid/data:/app/postpaid/data \
  -v $(pwd)/postpaid/models:/app/postpaid/models \
  pln-ml-trainer \
  bash -c "
    python postpaid/training/train_anomaly_score.py &&
    python postpaid/training/train_is_anomaly.py &&
    python postpaid/training/train_final_score_rumus1.py &&
    python postpaid/training/train_score_rumus2.py &&
    python postpaid/training/train_final_score_rumus2.py
  "
```

## Testing Worker

### 1. Start RabbitMQ (jika perlu)
```bash
docker run -d --name pln-rabbitmq \
  -p 5672:5672 -p 15672:15672 \
  rabbitmq:3-management
```

### 2. Start Worker
```bash
docker run -d \
  --name pln-ml-worker \
  -e RABBITMQ_HOST=host.docker.internal \
  --env-file .env \
  pln-ml-worker
```

### 3. Publish Test Message

```python
import pika
import json

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost')
)
channel = connection.channel()

message = {
    "job_id": "test_docker_001",
    "type": "prepaid",
    "features": {
        "avg_12m_pemakaian": 150.5,
        "std_12m_pemakaian": 25.3,
        "freq_tx_12m": 12,
        "avg_gap_days_12m": 30.5,
        "consecutive_anomaly_count": 3,
        "pemakaian": 145.0,
        "avg_pemakaian_gardu": 140.0
    }
}

channel.basic_publish(
    exchange='ml_exchange',
    routing_key='ml_inference_key',
    body=json.dumps(message)
)

print("Sent test message")
connection.close()
```

### 4. Check Logs
```bash
docker logs -f pln-ml-worker
```

## Troubleshooting

### Worker tidak bisa connect ke RabbitMQ

Jika pakai Docker Desktop, gunakan `host.docker.internal`:
```bash
docker run -d \
  --name pln-ml-worker \
  -e RABBITMQ_HOST=host.docker.internal \
  pln-ml-worker
```

### Model tidak ditemukan

Pastikan:
1. Training sudah dijalankan
2. Model files exist di `postpaid/models/v1/` atau `prepaid/models/v1/`

### Rebuild setelah code changes

```bash
docker build -t pln-ml-worker . --no-cache
docker build -f Dockerfile.trainer -t pln-ml-trainer . --no-cache
```

## Production Tips

### Persist Models

Volume mount untuk models:
```bash
docker run -d \
  -v $(pwd)/models:/app/models \
  pln-ml-worker
```

### Worker Restart Policy

```bash
docker run -d \
  --name pln-ml-worker \
  --restart unless-stopped \
  pln-ml-worker
```

### Multiple Workers

```bash
docker run -d --name worker-1 pln-ml-worker
docker run -d --name worker-2 pln-ml-worker
docker run -d --name worker-3 pln-ml-worker
```

### Resource Limits

```bash
docker run -d \
  --name pln-ml-worker \
  --memory="1g" \
  --cpus="1.0" \
  pln-ml-worker
```
