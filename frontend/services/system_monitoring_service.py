import sys
sys.dont_write_bytecode = True

import threading
import time
import psutil
from collections import deque
from dataclasses import dataclass
from typing import List

@dataclass
class MonitoringSnapshot:
    cpu: List[List[float]]
    ram: List[float]
    time: List[float]
    timestamp: str

class SystemMonitor:
    def __init__(self, update_interval: float = 1.0, history_size: int = 30):
        self.update_interval = update_interval
        self.lock = threading.Lock()
        self.cpu_history = deque(maxlen=history_size)
        self.ram_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)

        self.running = False
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return False  # Already running
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def reset(self):
        with self.lock:
            self.cpu_history.clear()
            self.ram_history.clear()
            self.time_history.clear()

    def _monitor_loop(self):
        while self.running:
            cpu = psutil.cpu_percent(percpu=True)
            ram = psutil.virtual_memory().percent
            timestamp = time.time()

            with self.lock:
                self.cpu_history.append(cpu)
                self.ram_history.append(ram)
                self.time_history.append(timestamp)

            time.sleep(self.update_interval)

    def get_snapshot(self) -> MonitoringSnapshot:
        with self.lock:
            return MonitoringSnapshot(
                cpu=list(self.cpu_history),
                ram=list(self.ram_history),
                time=list(self.time_history),
                timestamp=time.strftime('%H:%M:%S')
            )

# Singleton instance
_monitor = None

def get_monitor():
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor

def initialize_monitoring():
    return get_monitor().start()

def stop_monitoring():
    global _monitor
    if _monitor is not None:
        _monitor.stop()
        _monitor = None
