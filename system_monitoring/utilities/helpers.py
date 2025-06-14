import sys
sys.dont_write_bytecode = True

from functools import lru_cache
from typing import Tuple, Dict, Any

# Constants
KB = 1024
MB = KB * 1024
GB = MB * 1024
TB = GB * 1024

# Binary-based conversion units
UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

# Temperature thresholds
TEMP_CRITICAL = 80
TEMP_HIGH = 70
TEMP_WARM = 60

# Define constants for cleaner code
VOLTAGE_PI_RANGES = {
    'critical_high': 5.1,  # Too high - dangerous
    'optimal_high': 4.95,  # Optimal
    'warning': 4.8,        # Warning - getting low
}

VOLTAGE_BATTERY_RANGES = {
    'good': 4.5,    # Full/High (80-100%)
    'moderate': 4.0,  # Good (60-80%)
    'low': 3.5,     # Medium (30-60%)
}

RPM_RANGES = {
    'max': 8000,
    'high': 7000,
    'medium_high': 4000,
    'medium': 2500,
    'low': 1000,
    'off': 0
}

@lru_cache(maxsize=1024)
def format_disk_and_ram_values(value: float, from_unit: str) -> str:
    """Convert storage values to human-readable format with proper unit scaling."""
    try:
        unit_index = UNITS.index(from_unit.upper())
    except ValueError:
        raise ValueError(f"Invalid unit: {from_unit}. Use {UNITS}")

    while value >= 1024 and unit_index < len(UNITS) - 1:
        value /= 1024
        unit_index += 1
        
    return f"{value:.2f} {UNITS[unit_index]}"

@lru_cache(maxsize=1024)
def format_bytes(num_bytes: float) -> str:
    """Convert bytes to human-readable format (KB, MB, GB, TB).
    
    Args:
        num_bytes: Number of bytes to format
        
    Returns:
        Formatted string with appropriate unit
    """
    if num_bytes >= TB:
        return f"{num_bytes / TB:.2f} TB"
    elif num_bytes >= GB:
        return f"{num_bytes / GB:.2f} GB"
    elif num_bytes >= MB:
        return f"{num_bytes / MB:.2f} MB"
    elif num_bytes >= KB:
        return f"{num_bytes / KB:.2f} KB"
    else:
        return f"{num_bytes} B"

@lru_cache(maxsize=100)
def get_temp_indicator(temp: float) -> str:
    """Return icon and warning based on temperature thresholds.
    
    Args:
        temp: Temperature in Celsius
        
    Returns:
        String with appropriate temperature indicator icons
    """
    if temp >= TEMP_CRITICAL:
        return "🔴 🥵🔥"
    elif temp >= TEMP_HIGH:
        return "🟠 ⚠️"
    elif temp >= TEMP_WARM:
        return "🟠"
    else:
        return "⚪"

def format_temp_info(temp: float) -> str:
    """Format temperature info string with indicator.
    
    Args:
        temp: Temperature in Celsius
        
    Returns:
        Formatted temperature string with indicator
    """
    indicator = get_temp_indicator(temp)
    return f" | {indicator} {temp:.1f}°C"

@lru_cache(maxsize=100)
def get_fan_icon(rpm: int) -> str:
    """Return fan icon string based on RPM value.
    
    Args:
        rpm: Fan speed in RPM
        
    Returns:
        String with appropriate fan icons
    """
    if rpm > RPM_RANGES['high']:
        return "⏲⏲⏲⏲⏲"
    elif rpm > RPM_RANGES['medium_high']:
        return "⏲⏲⏲⏲"
    elif rpm > RPM_RANGES['medium']:
        return "⏲⏲⏲"
    elif rpm > RPM_RANGES['low']:
        return "⏲⏲"
    elif rpm > RPM_RANGES['off']:
        return "⏲"
    else:
        return "⛔"

@lru_cache(maxsize=1000)
def normalize_rpm_to_progress(rpm: int, max_rpm: int = 8000) -> int:
    """Normalize RPM value to a 0-100 scale for progress bars.
    
    Args:
        rpm: Fan speed in RPM
        max_rpm: Maximum expected RPM
        
    Returns:
        Integer value between 0-100 for progress bar
    """
    return min(int((rpm / max_rpm) * 100), 100)

def get_cpu_temp_alert(cpu_temps: list[float]) -> Tuple[str, str]:
    """
    Determine alert level and message based on CPU core temperatures.
    
    Args:
        cpu_temps: List of CPU core temperatures
        
    Returns:
        Tuple: (alert_type, message)
        alert_type: one of 'error', 'warning', 'info'
    """
    if not cpu_temps:
        return ("info", "")
        
    # More efficient check using any() and generator expression
    if any(t >= TEMP_CRITICAL for t in cpu_temps):
        return ("error", "🥵🔥 One or more CPU core(s) is/are running critically hot! 🔥🥵")
    elif any(t >= TEMP_HIGH for t in cpu_temps):
        return ("warning", "⚠️ One or more CPU core(s) is/are running hot!")
    return ("info", "")

def format_uptime(uptime: Dict[str, int]) -> str:
    """Format uptime dictionary into a human-readable string.
    
    Args:
        uptime: Dictionary containing hours, minutes, seconds
        
    Returns:
        Formatted uptime string
    """
    if not uptime:
        return "0h 0m 0s"
    return f"{uptime.get('hours', 0)}h {uptime.get('minutes', 0)}m {uptime.get('seconds', 0)}s"

@lru_cache(maxsize=100)
def get_voltage_icon(voltage_value: float, is_raspberry_pi: bool = True) -> str:
    """Return voltage icon string based on voltage value.
    
    Args:
        voltage_value: The voltage value to evaluate
        is_raspberry_pi: Whether this is for a Raspberry Pi (different thresholds)
    
    Returns:
        String of lightning bolt icons representing voltage level
    """
    if is_raspberry_pi:
        # Raspberry Pi specific thresholds (typically 4.8V - 5.2V range)
        if voltage_value > VOLTAGE_PI_RANGES['critical_high']:
            return "🗲🗲🗲🗲"
        elif voltage_value > VOLTAGE_PI_RANGES['optimal_high']:
            return "🗲🗲🗲"
        elif voltage_value > VOLTAGE_PI_RANGES['warning']:
            return "🗲🗲"
        else:
            return "🗲"
    else:
        # Generic system thresholds for battery-derived voltage (3.0V - 5.0V range)
        if voltage_value > VOLTAGE_BATTERY_RANGES['good']:
            return "🗲🗲🗲🗲"
        elif voltage_value > VOLTAGE_BATTERY_RANGES['moderate']:
            return "🗲🗲🗲"
        elif voltage_value > VOLTAGE_BATTERY_RANGES['low']:
            return "🗲🗲"
        else:
            return "🗲"

@lru_cache(maxsize=100)
def get_voltage_indicator(volts_value: float, is_raspberry_pi: bool = True) -> str:
    """Return icon and warning based on voltage thresholds.
    
    Args:
        volts_value: The voltage value to evaluate
        is_raspberry_pi: Whether this is for a Raspberry Pi (different thresholds)
    
    Returns:
        Colored circle indicator representing voltage status
    """
    if is_raspberry_pi:
        # Raspberry Pi specific thresholds
        if volts_value > VOLTAGE_PI_RANGES['critical_high']:
            return "🔴"  # Too high - dangerous
        elif volts_value > VOLTAGE_PI_RANGES['optimal_high']:
            return "🟢"  # Optimal
        elif volts_value > VOLTAGE_PI_RANGES['warning']:
            return "🟠"  # Warning - getting low
        else:
            return "⚫"  # Danger - too low
    else:
        # Generic system thresholds for battery-derived voltage
        if volts_value > VOLTAGE_BATTERY_RANGES['good']:
            return "🟢"  # Full/High (80-100%)
        elif volts_value > VOLTAGE_BATTERY_RANGES['moderate']:
            return "🟢"  # Good (60-80%)
        elif volts_value > VOLTAGE_BATTERY_RANGES['low']:
            return "🟠"  # Medium (30-60%)
        else:
            return "🔴"  # Low (0-30%)

def format_voltage_info(volts_value: float, is_raspberry_pi: bool = True) -> str:
    """Format voltage info string with indicator.
    
    Args:
        volts_value: The voltage value to format
        is_raspberry_pi: Whether this is for a Raspberry Pi (different thresholds)
    
    Returns:
        Formatted string with voltage indicators
    """
    icons = get_voltage_icon(volts_value, is_raspberry_pi)
    indicator = get_voltage_indicator(volts_value, is_raspberry_pi)
    return f" {indicator} {icons}"

def detect_power_source(power_info: Dict[str, Any]) -> str:
    """Detect what type of power source information is available.
    
    Args:
        power_info: The power info dictionary from the monitoring snapshot
        
    Returns:
        String indicating power source type: 'raspberry_pi', 'battery', or 'unknown'
    """
    if not power_info:
        return 'unknown'
        
    if 'pi_throttled' in power_info:
        return 'raspberry_pi'
    elif 'battery_percent' in power_info:
        return 'battery'
    else:
        return 'unknown'


@lru_cache(maxsize=100)
def get_voltage_indicator(rail: str, value: float) -> str:
    safe_ranges = {
        "VDD_CORE_V": (0.7, 1.1),
        "EXT5V_V": (4.75, 5.25),
        "3V3_SYS_V": (3.2, 3.4),
        "1V8_SYS_V": (1.7, 1.9),
        "1V1_SYS_V": (1.05, 1.15),
        "DDR_VDD2_V": (1.0, 1.2),
        "DDR_VDDQ_V": (0.55, 0.65),
        "3V3_ADC_V": (3.2, 3.4),
        "3V3_DAC_V": (3.2, 3.4),
        "HDMI_V": (4.75, 5.25),
        "0V8_AON_V": (0.7, 0.9),
        "0V8_SW_V": (0.7, 0.9),
        "3V7_WL_SW_V": (3.6, 3.8),
    }
    default_range = (value * 0.95, value * 1.05)
    low, high = safe_ranges.get(rail, default_range)
    if value == 0:
        return "⚠️"
    elif value < low * 0.98 or value > high * 1.02:
        return "🔴"
    elif value < low or value > high:
        return "🟡"
    else:
        return "🟢"


