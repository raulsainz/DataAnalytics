from kafka import KafkaProducer
from kafka.errors import KafkaError
from datetime import datetime
import psutil
import time
import json
kafka_broker='ncirl-az02.westeurope.cloudapp.azure.com:9092' 

producer = KafkaProducer(
    bootstrap_servers = [kafka_broker],
    value_serializer = lambda m: json.dumps(m).encode('utf-8')
)
i=0
while True :
    msg = 'Hola {}'.format(i)
    future = producer.send('cpu',msg)
    print(msg)
    i = i+1
    time.sleep(5)