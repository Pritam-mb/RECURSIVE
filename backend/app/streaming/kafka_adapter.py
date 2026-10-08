from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class KafkaAdapter:
    """Optional Kafka publisher for snapshots and alerts."""

    def __init__(self):
        self.enabled = os.getenv("KAFKA_ENABLED", "0") == "1"
        self.bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        self.snapshot_topic = os.getenv("KAFKA_TOPIC_SNAPSHOT", "orbital.snapshots")
        self.alerts_topic = os.getenv("KAFKA_TOPIC_ALERTS", "orbital.alerts")
        self._producer = None
        self._init_producer()

    def _init_producer(self):
        if not self.enabled:
            return

        try:
            from confluent_kafka import Producer

            self._producer = Producer({"bootstrap.servers": self.bootstrap})
            logger.info("Kafka producer initialized via confluent_kafka")
            return
        except Exception:
            self._producer = None

        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(bootstrap_servers=self.bootstrap)
            logger.info("Kafka producer initialized via kafka-python")
            return
        except Exception as error:
            logger.warning("Kafka producer unavailable: %s", error)
            self.enabled = False

    def _send(self, topic: str, payload: dict[str, Any] | str):
        if not self.enabled or self._producer is None:
            return

        message = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
        try:
            if hasattr(self._producer, "produce"):
                self._producer.produce(topic, message.encode("utf-8"))
                self._producer.poll(0)
            else:
                self._producer.send(topic, message.encode("utf-8"))
        except Exception as error:
            logger.warning("Kafka publish failed (%s): %s", topic, error)

    def publish_snapshot(self, payload: dict[str, Any] | str):
        self._send(self.snapshot_topic, payload)

    def publish_alerts(self, payload: dict[str, Any] | str):
        self._send(self.alerts_topic, payload)
