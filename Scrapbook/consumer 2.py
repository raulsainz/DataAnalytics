from kafka import KafkaConsumer
import json

kafka_broker='ncirl-az02.westeurope.cloudapp.azure.com:9092' 

consumer = KafkaConsumer('cpu',
                         bootstrap_servers = [kafka_broker],
                         auto_offset_reset = 'earliest',
                         group_id = None,
                         consumer_timeout_ms = 10000)
for message in consumer:
    print(message.value.decode())
