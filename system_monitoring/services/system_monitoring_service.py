import sys
sys.dont_write_bytecode = True

import threading
import time
import psutil
import subprocess
import platform
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Constants
HISTORY_SIZE_DEFAULT = 30
UPDATE_INTERVAL_DEFAULT = 1.0

# Use conditional import with typing
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

@dataclass
class MonitoringSnapshot:
    """Data class for system monitoring information snapshot."""
    cpu: List[List[float]]
    ram: List[float]
    time: List[float]
    timestamp: str
    disk: Dict[str, Any]
    network: Dict[str, Any]
    swap: Dict[str, Any]
    uptime: Dict[str, int]
    gpus: List[Dict[str, Any]]
    ram_info: Dict[str, float]
    cpu_temps: List[float]
    fan_speeds: Dict[str, List[float]]
    power_info: Dict[str, Any]  # includes all power-related info including voltage history

class SystemMonitor:
    """System resource monitoring class with threaded data collection."""
    
    def __init__(self, update_interval: float = UPDATE_INTERVAL_DEFAULT, history_size: int = HISTORY_SIZE_DEFAULT):
        """Initialize the system monitor.

        Args:
            update_interval: Time between updates in seconds
            history_size: Number of historical data points to keep
        """
        self.update_interval = update_interval
        self.lock = threading.RLock()  # Use RLock instead of Lock for nested locking
        self.cpu_history = deque(maxlen=history_size)
        self.ram_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.voltage_history = deque(maxlen=history_size)

        # Platform detection - do once at init
        self.is_raspberry_pi = self._is_raspberry_pi()
        self.battery_available = self._check_battery_available()

        # Thread control
        self.running = False
        self.thread = None

    @lru_cache(maxsize=1)
    def _is_raspberry_pi(self) -> bool:
        """Detect if running on a Raspberry Pi.

        Returns:
            bool: True if running on Raspberry Pi, False otherwise
        """
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Raspberry Pi' in model
        except (IOError, FileNotFoundError):
            try:
                # Alternative detection method
                output = subprocess.check_output(['vcgencmd', 'version'], stderr=subprocess.DEVNULL, timeout=1.0)
                return True
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                return False

    @lru_cache(maxsize=1)
    def _check_battery_available(self) -> bool:
        """Check if battery monitoring is available.

        Returns:
            bool: True if battery data is available, False otherwise
        """
        try:
            battery = psutil.sensors_battery()
            return battery is not None
        except (AttributeError, IOError, OSError):
            return False

    def start(self) -> bool:
        """Start the monitoring thread.

        Returns:
            bool: True if started successfully, False if already running
        """
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False  # Already running
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            return True

    def stop(self) -> None:
        """Stop the monitoring thread safely."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None

    def reset(self) -> None:
        """Clear all historical data."""
        with self.lock:
            self.cpu_history.clear()
            self.ram_history.clear()
            self.time_history.clear()
            self.voltage_history.clear()

    def _monitor_loop(self) -> None:
        """Main monitoring loop that collects system data periodically."""
        last_update_time = 0
        
        while self.running:
            current_time = time.time()
            # Ensure we maintain the update interval even if processing takes time
            time_to_sleep = max(0, self.update_interval - (current_time - last_update_time))
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            try:
                # Collect CPU and RAM data more efficiently
                cpu = psutil.cpu_percent(interval=None, percpu=True)  # Non-blocking
                ram = psutil.virtual_memory().percent
                timestamp = time.time()

                # Collect power data in each cycle
                self._collect_real_time_power_data()

                with self.lock:
                    self.cpu_history.append(cpu)
                    self.ram_history.append(ram)
                    self.time_history.append(timestamp)
            except Exception as e:
                # Log error and continue - don't crash monitoring thread
                print(f"Error in monitoring loop: {str(e)}")
                
            last_update_time = time.time()

    def _collect_real_time_power_data(self) -> None:
        """Collect all power-related metrics for real-time monitoring."""
        # Raspberry Pi voltage
        if self.is_raspberry_pi:
            try:
                all_voltages = self._get_all_pi_voltages()
                with self.lock:
                    self.voltage_history.append(all_voltages)
            except Exception:
                pass
            # try:
            #     # Use timeout to prevent blocking
            #     volts = subprocess.check_output(
            #         ["vcgencmd", "measure_volts", "core"],
            #         #["vcgencmd", "pmic_read_adc", "EXT5V_V"],
            #         stderr=subprocess.DEVNULL,
            #         timeout=0.5
            #     ).decode().strip()
            #     voltage_value = float(volts.split('=')[1].replace('V', ''))
            #     with self.lock:
            #         self.voltage_history.append(voltage_value)
            # except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            #     pass
        
        # Battery info (for laptops/WSL2)
        if self.battery_available:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    # Use battery percentage as a proxy for voltage on non-Pi systems
                    with self.lock:
                        # Convert battery percentage to a voltage-like value (e.g., 3.0V to 5.0V range)
                        simulated_voltage = 3.0 + (battery.percent / 100 * 2.0)
                        self.voltage_history.append(simulated_voltage)
            except (AttributeError, OSError):
                pass

    def _get_all_pi_voltages(self) -> dict:
        """Collect all voltage rails available on Raspberry Pi 5."""
        voltages = {}
        try:
            output = subprocess.check_output(
                ["sudo", "vcgencmd", "pmic_read_adc"],
                timeout=1.0
            ).decode().strip()
            for line in output.split('\n'):
                if 'volt(' in line and '_V' in line:
                    rail = line.split()[0]  # e.g., "VDD_CORE_V"
                    value = float(line.split('=')[1].replace('V', ''))
                    if isinstance(value, (int, float)):
                        voltages[rail] = value
        except Exception as e:
            pass
        return voltages

    def get_snapshot(self) -> MonitoringSnapshot:
        """Create a complete snapshot of the current system state.
        
        Returns:
            MonitoringSnapshot: Dataclass containing all system metrics
        """
        with self.lock:
            # Copy history data to avoid modification during processing
            cpu_history = list(self.cpu_history)
            ram_history = list(self.ram_history)
            time_history = list(self.time_history)
            
            # Power (collect the rest of power info that's not needed for real-time tracking)
            power_info = self._collect_power_info()
            
            try:
                # Disk - only check the root filesystem
                disk = psutil.disk_usage('/')
                disk_data = {
                    'percent': disk.percent,
                    'used': round(disk.used / (1024**3), 2),
                    'total': round(disk.total / (1024**3), 2)
                }
            except OSError:
                disk_data = {'percent': 0, 'used': 0, 'total': 0}

            try:
                # Network
                net = psutil.net_io_counters()
                net_data = {
                    'sent': round(net.bytes_sent / (1024**2), 2),
                    'recv': round(net.bytes_recv / (1024**2), 2)
                }
            except OSError:
                net_data = {'sent': 0, 'recv': 0}

            try:
                # Swap
                swap = psutil.swap_memory()
                swap_data = {
                    'percent': swap.percent,
                    'used': round(swap.used / (1024**3), 2),
                    'total': round(swap.total / (1024**3), 2)
                }
            except OSError:
                swap_data = {'percent': 0, 'used': 0, 'total': 0}

            try:
                # RAM Info
                ram_mem = psutil.virtual_memory()
                ram_info = {
                    'used': round(ram_mem.used / (1024**3), 2),
                    'total': round(ram_mem.total / (1024**3), 2),
                }
            except OSError:
                ram_info = {'used': 0, 'total': 0}

            try:
                # Uptime
                uptime_sec = time.time() - psutil.boot_time()
                hours, rem = divmod(uptime_sec, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_data = {
                    'hours': int(hours),
                    'minutes': int(minutes),
                    'seconds': int(seconds)
                }
            except OSError:
                uptime_data = {'hours': 0, 'minutes': 0, 'seconds': 0}

            # GPU - collect only if available
            gpu_data = self._collect_gpu_data()

            # CPU temperatures and fan speeds
            cpu_temps = self._collect_cpu_temps()
            fan_speeds = self._collect_fan_speeds()

            return MonitoringSnapshot(
                cpu=cpu_history,
                ram=ram_history,
                time=time_history,
                timestamp=time.strftime('%H:%M:%S'),
                disk=disk_data,
                network=net_data,
                swap=swap_data,
                uptime=uptime_data,
                gpus=gpu_data,
                ram_info=ram_info,
                cpu_temps=cpu_temps,
                fan_speeds=fan_speeds,
                power_info=power_info
            )
    
    def _collect_gpu_data(self) -> List[Dict[str, Any]]:
        """Collect GPU information if available.
    
        Returns:
            List of dictionaries with GPU metrics
        """
        gpu_data = []
        if GPU_AVAILABLE:
            try:
                for gpu in GPUtil.getGPUs():
                    gpu_data.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': round(gpu.load * 100),
                        'mem_used': gpu.memoryUsed,
                        'mem_total': gpu.memoryTotal,
                        'temp': gpu.temperature
                    })
            except Exception:
                pass
        return gpu_data
    
    def _collect_cpu_temps(self) -> List[float]:
        """Collect CPU temperature information.

        Returns:
            List of CPU core temperatures
        """
        try:
            temps = psutil.sensors_temperatures()
            # Try different sensor names based on platform
            for sensor_name in ["cpu_thermal", "coretemp", "cpu-thermal", "k10temp"]:
                if sensor_name in temps:
                    return [
                        entry.current
                        for entry in temps[sensor_name]
                        if "Core" in entry.label or not entry.label
                    ]
            return []
        except (AttributeError, OSError):
            return []
    
    def _collect_fan_speeds(self) -> Dict[str, List[float]]:
        """Collect fan speed information.

        Returns:
            Dictionary mapping fan controllers to lists of speeds
        """
        try:
            fans = psutil.sensors_fans()
            return {
                name: [fan.current for fan in entries]
                for name, entries in fans.items()
            }
        except (AttributeError, OSError):
            return {}

    def _collect_power_info(self) -> Dict[str, Any]:
        """Collect all power-related information.

        Returns:
            Dictionary with power metrics
        """
        power_data = {}

        # Battery info (for laptops/WSL2)
        if self.battery_available:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    power_data['battery_percent'] = battery.percent
                    power_data['power_plugged'] = battery.power_plugged
                    power_data['secs_left'] = battery.secsleft
            except (AttributeError, OSError):
                pass

        # Raspberry Pi voltage and throttling
        if self.is_raspberry_pi:
            try:
                # Get voltage info
                volts = subprocess.check_output(
                    ["vcgencmd", "measure_volts", "core"],
                    stderr=subprocess.DEVNULL,
                    timeout=0.5
                ).decode().strip()

                voltage_value = float(volts.split('=')[1].replace('V', ''))
                power_data['pi_voltage'] = volts
                power_data['pi_voltage_val'] = voltage_value

                # Check for throttling status
                throttled = subprocess.check_output(
                    ["vcgencmd", "get_throttled"],
                    stderr=subprocess.DEVNULL,
                    timeout=0.5
                ).decode().strip()

                power_data['pi_throttled'] = throttled
            except (subprocess.SubprocessError, ValueError, IndexError, OSError):
                pass

        if self.is_raspberry_pi:
            try:
                all_voltages = self._get_all_pi_voltages()
                power_data['pi_all_voltages'] = all_voltages
                # Optionally, add the most important rail as 'pi_voltage_val'
                if 'VDD_CORE_V' in all_voltages:
                    power_data['pi_voltage_val'] = all_voltages['VDD_CORE_V']
            except Exception:
                pass

        # Always include voltage history
        with self.lock:
            power_data['voltage_history'] = list(self.voltage_history)

        # Add a current voltage value if history exists but no current value
        if power_data.get('voltage_history') and 'pi_voltage_val' not in power_data:
            power_data['pi_voltage_val'] = power_data['voltage_history'][-1]
            # Create a display string if on non-Pi systems
            if not self.is_raspberry_pi and self.battery_available:
                power_data['pi_voltage'] = f"volt={power_data['pi_voltage_val']:.2f}V"

        return power_data


# Singleton instance management with thread safety
_monitor_lock = threading.Lock()
_monitor_instance: Optional[SystemMonitor] = None

def get_monitor(update_interval: float = UPDATE_INTERVAL_DEFAULT) -> SystemMonitor:
    """Get or create the singleton SystemMonitor instance.

    Args:
        update_interval: Time between updates in seconds
        
    Returns:
        SystemMonitor: The singleton instance
    """
    global _monitor_instance
    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = SystemMonitor(update_interval=update_interval)
        return _monitor_instance

def initialize_monitoring(update_interval: float = UPDATE_INTERVAL_DEFAULT) -> bool:
    """Initialize and start the monitoring system.

    Args:
        update_interval: Time between updates in seconds
        
    Returns:
        bool: True if started successfully, False otherwise
    """
    monitor = get_monitor(update_interval)
    return monitor.start()

def stop_monitoring() -> None:
    """Stop the monitoring system safely."""
    global _monitor_instance
    with _monitor_lock:
        if _monitor_instance is not None:
            _monitor_instance.stop()
            # Don't set to None to allow reuse of the same instance