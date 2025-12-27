# RabbitMQ Worker for PLN ML System

Worker untuk memproses inference tasks dari RabbitMQ queue dan menjalankan pipeline postpaid/prepaid.

## Prerequisites

```bash
# Required
pip install pika

# Optional (for .env file support)
pip install python-dotenv
```

## Configuration

Semua configuration bisa di-set via environment variables atau menggunakan file `.env`:

### Method 1: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_HOST` | localhost | RabbitMQ host |
| `RABBITMQ_PORT` | 5672 | RabbitMQ port |
| `RABBITMQ_USER` | guest | RabbitMQ username |
| `RABBITMQ_PASS` | guest | RabbitMQ password |
| `RABBITMQ_URL` | None | Full RabbitMQ URL (overrides host/port/user/pass) |
| `RABBITMQ_QUEUE` | ml_inference_queue | Queue name |
| `RABBITMQ_EXCHANGE` | ml_exchange | Exchange name |
| `RABBITMQ_EXCHANGE_TYPE` | direct | Exchange type |
| `RABBITMQ_ROUTING_KEY` | ml_inference_key | Routing key |
| `RABBITMQ_HEARTBEAT` | 600 | Heartbeat interval (seconds) |

### Method 2: Using .env File (Recommended)

1. Copy `.env.example` to project root:
```bash
cp worker/.env.example .env
```

2. Edit `.env` file in project root with your configuration:
```bash
RABBITMQ_HOST=your-rabbitmq-host
RABBITMQ_PORT=5672
RABBITMQ_USER=your-username
RABBITMQ_PASS=your-password
```

3. Load environment variables before running worker:
```bash
# Using python-dotenv (recommended)
pip install python-dotenv

# Run with dotenv from project root
python -m dotenv run python worker/consume.py

# Or manually export from project root
export $(cat .env | xargs)
python worker/consume.py
```

## Running the Worker

```bash
# From project root
python worker/consume.py
```

## Message Format

### Prepaid Inference

```json
{
  "job_id": "job_123",
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
```

### Postpaid Inference

```json
{
  "job_id": "job_456",
  "batch_id": "batch_789",
  "type": "postpaid",
  "features": {
    "pemakaian": 41.0,
    "baseline": 40.0,
    "avg_pemakaian_gardu": 60.0,
    "threshold_drop_consume": 30.0,
    "consecutive_anomaly_count": 2
  }
}
```

## Testing

### 1. Start RabbitMQ (Docker)

```bash
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management
```

### 2. Publish Test Message

Python script untuk test:

```python
import pika
import json

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost')
)
channel = connection.channel()

# Declare exchange and queue
channel.exchange_declare(exchange='ml_exchange', exchange_type='direct', durable=True)
channel.queue_declare(queue='ml_inference_queue', durable=True)
channel.queue_bind(exchange='ml_exchange', queue='ml_inference_queue', routing_key='ml_inference_key')

# Publish prepaid message
prepaid_payload = {
    "job_id": "test_job_001",
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
    body=json.dumps(prepaid_payload),
    properties=pika.BasicProperties(delivery_mode=2)  # persistent
)

print(" [x] Sent prepaid message")

connection.close()
```

## Architecture

```
RabbitMQ Queue
    ↓
Worker (consume.py)
    ↓
Parse Payload (type: prepaid/postpaid)
    ↓
Run Pipeline (run_pipeline)
    ↓
Log Results
    ↓
ACK Message
```

## Key Changes from Old Version

1. **Removed**: SQL processor & database queries
2. **Removed**: Complex model loading logic
3. **Added**: Direct `run_pipeline()` call with features from JSON
4. **Added**: Feature validation before processing
5. **Simplified**: All config via environment variables
6. **Fixed**: Import paths to match new project structure

## Error Handling

- Missing required features → Job rejected, message NACKed
- Pipeline execution error → Job rejected, message NACKed
- Invalid JSON → Message rejected
- Unknown job type → Message rejected

## Logs

Worker akan log:
- Job ID yang diproses
- Features yang diterima
- Hasil inference (predictions/summary)
- Error details dengan full traceback
