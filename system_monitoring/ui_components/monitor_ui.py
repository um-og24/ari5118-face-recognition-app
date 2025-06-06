import sys
sys.dont_write_bytecode = True

import streamlit as st
import functools
from typing import Dict, Tuple, Any
from services.system_monitoring_service import MonitoringSnapshot

from utilities.helpers import (
    format_bytes,
    format_disk_and_ram_values,
    format_temp_info,
    get_cpu_temp_alert,
    get_fan_icon,
    normalize_rpm_to_progress,
    format_uptime,
    detect_power_source,
    format_voltage_info,
    get_voltage_icon,
    get_voltage_indicator,
)

# Add caching to expensive UI operations
@functools.lru_cache(maxsize=32)
def create_section_title(icon: str, title: str) -> str:
    """Create a section title with icon for consistent UI.
    
    Args:
        icon: Icon to display before title
        title: Section title text
        
    Returns:
        Formatted section title string
    """
    return f"{icon} {title}"

# Partition the UI drawing into smaller, reusable components
def draw_monitoring_dashboard(snapshot: MonitoringSnapshot, container):
    """Render full system dashboard using collected snapshot data.
    
    Args:
        snapshot: Current monitoring data snapshot
        container: Streamlit container to render in
    """
    with container.container():
        # Layout the dashboard sections
        draw_power_info(snapshot)
        draw_cpu_stats(snapshot)
        
        # First row of metrics
        col1, col2 = st.columns(2)
        with col1:
            draw_fan_speeds(snapshot)
        with col2:
            draw_disk_usage(snapshot)

        # Second row of metrics
        col1, col2 = st.columns(2)
        with col1:
            draw_ram_usage(snapshot)
        with col2:
            draw_swap_usage(snapshot)

        # GPU section (full width)
        draw_gpu_usage(snapshot)

        # Third row of metrics
        col1, col2 = st.columns(2)
        with col1:
            draw_network_usage(snapshot)
        with col2:
            draw_uptime(snapshot)

        # Timestamp at bottom
        draw_timestamp(snapshot)

# --- Section Functions ---

def draw_power_info(snapshot: MonitoringSnapshot):
    """Draw power information section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    st.subheader(create_section_title("⚡", "Power Info"))
    
    with st.container(border=True):
        power = snapshot.power_info
        if not power:
            st.text("No power information available")
            return

        # Detect power source type (do this only once)
        power_source = detect_power_source(power)
        is_raspberry_pi = (power_source == 'raspberry_pi')
        
        # Battery percentage & status (for laptops/desktops)
        if power_source == 'battery':
            _draw_battery_info(power)
        
        # Display voltage information (available on both Pi and non-Pi systems)
        if 'pi_voltage' in power or 'voltage_history' in power:
            _draw_voltage_info(power, power_source, is_raspberry_pi)

def _draw_battery_info(power: Dict[str, Any]):
    """Helper function to draw battery information.
    
    Args:
        power: Power information dictionary
    """
    plugged = "🔌 AC" if power.get('power_plugged') else "🔋 Battery"
    warning = ""
    if not power.get('power_plugged') and power['battery_percent'] <= 25:
        warning = " ⚠️ 🔋⚡ Battery is low! Consider plugging in power. ⚡🔋"
    
    # Battery remaining time
    time_info = ""
    secs_left = power.get('secs_left', -1)
    if secs_left > 0:
        hours, remainder = divmod(secs_left, 3600)
        minutes, _ = divmod(remainder, 60)
        time_info = f" | {int(hours)}h {int(minutes)}m remaining"
    elif secs_left == -1 and power.get('power_plugged'):
        time_info = " | Charging"
        
    st.text(f"Battery: {power['battery_percent']}% ({plugged}{time_info}){warning}")

def _draw_voltage_info_outdated(power: Dict[str, Any], power_source: str, is_raspberry_pi: bool):
    """Helper function to draw voltage information.

    Args:
        power: Power information dictionary
        power_source: Type of power source ('raspberry_pi', 'battery', 'unknown')
        is_raspberry_pi: Boolean indicating if we're on a Raspberry Pi
    """
    # Get voltage label and value
    volts_label = power.get('pi_voltage', 'N/A')
    volts_value = power.get('pi_voltage_val', 0)

    # Set appropriate title based on device type
    voltage_title = "Pi Voltage: " if is_raspberry_pi else "System Voltage: "

    # Format voltage info with appropriate context
    voltage_info = format_voltage_info(volts_value, is_raspberry_pi)

    # Add throttle info for Raspberry Pi
    if is_raspberry_pi:
        throttled = power.get('pi_throttled', "")
        if throttled and throttled != "throttled=0x0":
            voltage_info += f" ⚠️ Pi is throttled! ({throttled})"

    # Display voltage with appropriate context
    st.text(f"{voltage_title}{volts_label}{voltage_info}")

    # Min/Max/Current voltage summary
    vhist = power.get('voltage_history', [])
    if vhist:
        vmin = min(vhist)
        vmax = max(vhist)
        vcurrent = vhist[-1] if vhist else volts_value

        if is_raspberry_pi:
            st.caption(f"Recent: min {vmin:.2f}V | max {vmax:.2f}V | current {vcurrent:.2f}V")
        else:
            # For non-Pi systems, explain the voltage values if they're derived from battery
            if power_source == 'battery':
                st.caption(f"Voltage representation: min {vmin:.2f}V | max {vmax:.2f}V | current {vcurrent:.2f}V")
                st.caption("(Voltage values derived from battery percentage)")

def _draw_voltage_info(power: Dict[str, Any], power_source: str, is_raspberry_pi: bool):
    """Helper function to draw voltage information.

    Args:
        power: Power information dictionary
        power_source: Type of power source ('raspberry_pi', 'battery', 'unknown')
        is_raspberry_pi: Boolean indicating if we're on a Raspberry Pi
    """
    # Get voltage label and value
    all_voltages = power.get('pi_all_voltages', None)

    # Set appropriate title based on device type
    voltage_title = "Pi Voltages" if is_raspberry_pi else "System Voltage: "

    if is_raspberry_pi:
        throttled = power.get('pi_throttled', "")
        if throttled and throttled != "throttled=0x0":
            voltage_title += f" | ⚠️ Pi is throttled! ({throttled})"

    st.text(voltage_title)

    if all_voltages and isinstance(all_voltages, dict) and len(all_voltages) > 0:
        # Sort the voltage rails
        sorted_voltages = sorted(all_voltages.items(), key=lambda x: str(x[0]))
        # Calculate number of columns needed (you can adjust this based on preference)
        num_columns = min(5, len(sorted_voltages))  # Maximum 5 columns to avoid too much spreading
        # Create columns
        cols = st.columns(num_columns)

        #voltage_lines = []
        for i, (rail, value) in enumerate(sorted_voltages):
            col_index = i % num_columns
            with cols[col_index]:
                if isinstance(value, (int, float)):
                    icon = get_voltage_indicator(rail, value)
                    decor_icon = get_voltage_icon(value, is_raspberry_pi)
                    st.text(f"{icon} {rail}: {value:.3f}V {decor_icon}")

        #voltage_info = "\n".join(voltage_lines) if voltage_lines else "No numeric voltage rails found."
    else:
        # Get voltage label and value
        volts_label = power.get('pi_voltage', 'N/A')
        volts_value = power.get('pi_voltage_val', 0)

        # Format voltage info with appropriate context
        voltage_info = format_voltage_info(volts_value, is_raspberry_pi)
        voltage_info = f"{volts_label} ({volts_value:.4f}V) {voltage_info}"

        # Display voltage with appropriate context
        st.text(voltage_info)

    # Min/Max/Current voltage summary (for the main rail, if available)
    vhist = power.get('voltage_history', [])
    if vhist:
        core_voltages = [item.get('VDD_CORE_V', 0) if isinstance(item, dict) else 0 for item in vhist]
        core_voltages = [v for v in core_voltages if v != 0]
        if core_voltages:
            vmin = min(core_voltages)
            vmax = max(core_voltages)
            vcurrent = core_voltages[-1]
        else:
            vmin, vmax, vcurrent = 0, 0, 0

        if is_raspberry_pi:
            st.caption(f"Recent (core): min {vmin:.2f}V | max {vmax:.2f}V | current {vcurrent:.2f}V")
        else:
            if power_source == 'battery':
                st.caption(f"Voltage representation: min {vmin:.2f}V | max {vmax:.2f}V | current {vcurrent:.2f}V")
                st.caption("(Voltage values derived from battery percentage)")

def draw_cpu_stats(snapshot: MonitoringSnapshot):
    """Draw CPU core statistics section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    cpu = snapshot.cpu[-1] if snapshot.cpu else []
    temps = snapshot.cpu_temps or []

    # Create a header with indicator if temps are available
    col1, col2 = st.columns(2)
    with col1:
        temp_info = format_temp_info(temps[0]) if temps else ""
        st.subheader(create_section_title("🧮", f"CPU Core Stats{temp_info}"))
    with col2:
        alert_type, alert_msg = get_cpu_temp_alert(temps)
        if alert_type == "error":
            st.error(alert_msg)
        elif alert_type == "warning":
            st.warning(alert_msg)
        else:
            st.write("")

    # CPU usage progress bars
    with st.container(border=True):
        for idx, usage in enumerate(cpu):
            # Only show temp info for cores that have temperature readings
            core_temp_info = format_temp_info(temps[idx]) if idx < len(temps) else ""
            st.text(f"Core {idx}: {usage:.1f}%{core_temp_info}")
            st.progress(int(usage))

def draw_fan_speeds(snapshot: MonitoringSnapshot):
    """Draw fan speed section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    MAX_RPM = 8000  # Consider moving to helpers.py as a constant
    st.subheader(create_section_title("🌀", "Fan Speeds"))
    
    with st.container(border=True):
        if snapshot.fan_speeds:
            for name, speeds in snapshot.fan_speeds.items():
                for idx, rpm in enumerate(speeds):
                    icons = get_fan_icon(rpm)
                    bar_value = normalize_rpm_to_progress(rpm, MAX_RPM)
                    st.progress(bar_value)
                    st.text(f"Fan {idx} ({name}): {rpm} RPM {icons}")
        else:
            st.progress(0)
            display_a_centered_caption("💀🚫 _...unable to acquire statistics..._ 🚫💀")

def draw_disk_usage(snapshot: MonitoringSnapshot):
    """Draw disk usage section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    st.subheader(create_section_title("💾", "Disk Usage"))
    
    with st.container(border=True):
        disk = snapshot.disk
        if disk:
            st.progress(int(disk['percent']))
            # Convert from original unit (e.g., GB) to appropriate scaled unit
            disk_used = format_disk_and_ram_values(disk['used'], disk.get('unit', 'GB'))
            disk_total = format_disk_and_ram_values(disk['total'], disk.get('unit', 'GB'))
            st.text(f"Disk Usage: {disk['percent']:.1f}% ({disk_used} / {disk_total})")
        else:
            st.progress(0)
            st.text("Disk Usage: N/A")

def draw_ram_usage(snapshot: MonitoringSnapshot):
    """Draw RAM usage section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    ram = snapshot.ram[-1] if snapshot.ram else 0
    ram_info = snapshot.ram_info or {'used': 0, 'total': 0}
    
    st.subheader(create_section_title("🧠", "RAM Usage"))
    
    with st.container(border=True):
        st.progress(int(ram))
        ram_used=format_disk_and_ram_values(ram_info['used'], "GB")
        ram_total=format_disk_and_ram_values(ram_info['total'], "GB")
        st.text(f"RAM: {ram:.1f}% ({ram_used} / {ram_total})")

def draw_swap_usage(snapshot: MonitoringSnapshot):
    """Draw swap memory usage section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    st.subheader(create_section_title("🔃", "Swap Memory"))
    
    with st.container(border=True):
        swap = snapshot.swap or {'percent': 0, 'used': 0, 'total': 0}
        st.progress(int(swap['percent']))
        swap_used=format_disk_and_ram_values(swap['used'], "GB")
        swap_total=format_disk_and_ram_values(swap['total'], "GB")
        st.text(f"Swap: {swap['percent']:.1f}% ({swap_used} / {swap_total})")

def draw_gpu_usage(snapshot: MonitoringSnapshot):
    """Draw GPU usage section if GPUs are available.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    if snapshot.gpus:
        st.subheader(create_section_title("🎮", "GPU Usage"))
        
        with st.container(border=True):
            for gpu in snapshot.gpus:
                st.text(f"{gpu['name']} (GPU {gpu['id']})")
                st.progress(int(gpu['load']))
                st.caption(f"{gpu['mem_used']}MB / {gpu['mem_total']}MB | Temp: {gpu['temp']}°C")

def draw_network_usage(snapshot: MonitoringSnapshot):
    """Draw network I/O section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    st.subheader(create_section_title("🌐", "Network I/O"))
    
    with st.container(border=True):
        net = snapshot.network or {'sent': 0, 'recv': 0}
        # Pre-calculate the byte values rather than doing it inline
        sent_bytes = net['sent'] * 1024 * 1024  # Convert from MB to bytes
        recv_bytes = net['recv'] * 1024 * 1024
        sent_readable = format_bytes(sent_bytes)
        recv_readable = format_bytes(recv_bytes)
        st.text(f"Sent: {sent_readable} | Received: {recv_readable}")

def draw_uptime(snapshot: MonitoringSnapshot):
    """Draw system uptime section.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    st.subheader(create_section_title("🕒", "System Uptime"))
    
    with st.container(border=True):
        uptime = snapshot.uptime or {}
        st.text(f"Uptime: {format_uptime(uptime)}")

def draw_timestamp(snapshot: MonitoringSnapshot):
    """Draw last update timestamp at the bottom of the dashboard.
    
    Args:
        snapshot: Current monitoring data snapshot
    """
    _, col = st.columns([2, 1])
    with col:
        display_a_centered_caption(f"Last updated: {snapshot.timestamp}")

def display_a_centered_caption(text: str):
    """Display a centered caption text.
    
    Args:
        text: Text to display as caption
    """
    _,col,_=st.columns(3)
    with col:
        st.caption(text)