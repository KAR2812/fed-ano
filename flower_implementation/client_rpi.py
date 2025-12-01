import flwr as fl
import psutil
import time
import socket
import numpy as np

SERVER_ADDR = "10.10.20.162:8080"

# Dummy tiny model (replace with TFLite)
class TinyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [np.array([0.1, 0.2, 0.3])]  # dummy parameters

    def fit(self, parameters, config):
        start = time.time()

        mem = psutil.virtual_memory().used / (1024*1024)
        cpu = psutil.cpu_percent()

        # log metrics
        with open("hw_metrics.csv", "a") as f:
            f.write(f"{time.time()},{mem},{cpu}\n")

        time.sleep(1)
        print("Training completed on RPi-Sim")

        return parameters, 10, {}

    def evaluate(self, parameters, config):
        return 0.0, 10, {}

fl.client.start_numpy_client(SERVER_ADDR, TinyClient())
