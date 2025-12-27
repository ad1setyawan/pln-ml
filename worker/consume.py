#!/usr/bin/env python3
"""
RabbitMQ Worker for PLN Anomaly Detection ML System
Listens for inference tasks and triggers the appropriate pipeline.
"""

import sys
import json
import logging
import time
import os
import pika
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import pipeline functions
try:
    from postpaid.inferences.predict_pipeline import run_pipeline as postpaid_run_pipeline
    POSTPAID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Postpaid pipeline not available: {e}")
    POSTPAID_AVAILABLE = False

try:
    from prepaid.inferences.predict_pipeline import run_pipeline as prepaid_run_pipeline
    PREPAID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Prepaid pipeline not available: {e}")
    PREPAID_AVAILABLE = False

# RabbitMQ Configuration (can be overridden with environment variables)
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.getenv('RABBITMQ_PASS', 'guest')
RABBITMQ_URL = os.getenv('RABBITMQ_URL', None)
RABBITMQ_QUEUE = os.getenv('RABBITMQ_QUEUE', 'ml_inference_queue')
RABBITMQ_EXCHANGE = os.getenv('RABBITMQ_EXCHANGE', 'ml_exchange')
RABBITMQ_EXCHANGE_TYPE = os.getenv('RABBITMQ_EXCHANGE_TYPE', 'direct')
RABBITMQ_ROUTING_KEY = os.getenv('RABBITMQ_ROUTING_KEY', 'ml_inference_key')
RABBITMQ_HEARTBEAT = int(os.getenv('RABBITMQ_HEARTBEAT', 600))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("worker")


def process_prepaid_job(payload: Dict[str, Any]) -> bool:
    """
    Process prepaid inference job.

    Expected payload format:
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
    """
    if not PREPAID_AVAILABLE:
        logger.error("Prepaid pipeline not available")
        return False

    job_id = payload.get('job_id')
    features = payload.get('features', {})

    logger.info(f"Starting Prepaid inference for Job ID: {job_id}")

    try:
        # Validate required features
        required_features = [
            'avg_12m_pemakaian', 'std_12m_pemakaian', 'freq_tx_12m',
            'avg_gap_days_12m', 'consecutive_anomaly_count',
            'pemakaian', 'avg_pemakaian_gardu'
        ]

        missing = [f for f in required_features if f not in features]
        if missing:
            logger.error(f"Missing required features: {missing}")
            return False

        # Run pipeline
        results = prepaid_run_pipeline(features=features)

        # Log results
        logger.info(f"Prepaid job {job_id} completed successfully")
        logger.info(f"Results: {json.dumps(results.get('predictions', {}), indent=2)}")

        return True

    except Exception as e:
        logger.error(f"Error executing prepaid job {job_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def process_postpaid_job(payload: Dict[str, Any]) -> bool:
    """
    Process postpaid inference job.

    Expected payload format:
    {
        "job_id": "job_123",
        "batch_id": "batch_456",  # optional
        "type": "postpaid",
        "features": {
            "pemakaian": 41.0,
            "baseline": 40.0,
            "avg_pemakaian_gardu": 60.0,
            "threshold_drop_consume": 30.0,
            "consecutive_anomaly_count": 2
        }
    }
    """
    if not POSTPAID_AVAILABLE:
        logger.error("Postpaid pipeline not available")
        return False

    job_id = payload.get('job_id')
    batch_id = payload.get('batch_id')
    features = payload.get('features', {})

    logger.info(f"Starting Postpaid inference for Job ID: {job_id}, Batch ID: {batch_id}")

    try:
        # Validate required features
        required_features = [
            'pemakaian', 'baseline', 'avg_pemakaian_gardu',
            'threshold_drop_consume', 'consecutive_anomaly_count'
        ]

        missing = [f for f in required_features if f not in features]
        if missing:
            logger.error(f"Missing required features: {missing}")
            return False

        # Run pipeline
        results = postpaid_run_pipeline(features=features)

        # Log results
        logger.info(f"Postpaid job {job_id} completed successfully")
        logger.info(f"Results: {json.dumps(results.get('summary', {}), indent=2)}")

        return True

    except Exception as e:
        logger.error(f"Error executing postpaid job {job_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def callback(ch, method, _properties, body):
    """RabbitMQ message handler"""
    try:
        logger.info(f"Received message")
        payload = json.loads(body)

        # Check 'type' field
        job_type = payload.get('type')
        job_id = payload.get('job_id')

        if not job_type or not job_id:
            logger.error("Invalid payload: missing type or job_id")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return

        success = False
        if job_type.lower() == 'prepaid':
            success = process_prepaid_job(payload)
        elif job_type.lower() == 'postpaid':
            success = process_postpaid_job(payload)
        else:
            logger.error(f"Unknown job type: {job_type}")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return

        if success:
            logger.info(f"Job {job_id} processed successfully, acknowledging message")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            logger.error(f"Job {job_id} failed processing")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON payload")
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        logger.error(f"Unexpected error in callback: {e}")
        import traceback
        logger.error(traceback.format_exc())
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    """Main worker loop"""
    while True:
        try:
            if RABBITMQ_URL:
                logger.info("Connecting to RabbitMQ via URLParameters")
                parameters = pika.URLParameters(RABBITMQ_URL)
                if RABBITMQ_HEARTBEAT:
                     parameters.heartbeat = RABBITMQ_HEARTBEAT
            else:
                logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")
                credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
                parameters = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    credentials=credentials,
                    heartbeat=RABBITMQ_HEARTBEAT,
                    blocked_connection_timeout=300
                )

            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()

            # Declare exchange
            channel.exchange_declare(
                exchange=RABBITMQ_EXCHANGE,
                exchange_type=RABBITMQ_EXCHANGE_TYPE,
                durable=True
            )

            # Declare queue
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

            # Bind queue to exchange with routing key
            channel.queue_bind(
                exchange=RABBITMQ_EXCHANGE,
                queue=RABBITMQ_QUEUE,
                routing_key=RABBITMQ_ROUTING_KEY
            )

            # Prefetch count
            channel.basic_qos(prefetch_count=10)

            logger.info(f"Waiting for tasks in queue '{RABBITMQ_QUEUE}' bound to exchange '{RABBITMQ_EXCHANGE}' with key '{RABBITMQ_ROUTING_KEY}'")

            channel.basic_consume(
                queue=RABBITMQ_QUEUE,
                on_message_callback=callback
            )

            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Connection lost, retrying in 5 seconds... Error: {e}")
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(5)

if __name__ == "__main__":
    main()
