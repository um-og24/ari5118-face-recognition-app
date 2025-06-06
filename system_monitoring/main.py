import sys
sys.dont_write_bytecode = True

import streamlit as st
import time
import threading
from services.system_monitoring_service import initialize_monitoring, stop_monitoring, get_monitor
from ui_components.monitor_ui import draw_monitoring_dashboard

# Configuration constants
UPDATE_INTERVAL = 0.25  # seconds - reduced for more responsive UI updates
DEFAULT_HISTORY_SIZE = 30

# Create a lock for thread-safe access to session state
state_lock = threading.RLock()

def initialize_session_state():
    """Initialize or reset session state variables."""
    with state_lock:
        if 'monitoring_running' not in st.session_state:
            st.session_state.monitoring_running = False
        if 'monitoring_options' not in st.session_state:
            st.session_state.monitoring_options = {
                'update_interval': UPDATE_INTERVAL,
                'history_size': DEFAULT_HISTORY_SIZE
            }

def toggle_monitoring():
    """Toggle monitoring state based on UI toggle."""
    with state_lock:
        st.session_state.monitoring_running = not st.session_state.monitoring_running
        if st.session_state.monitoring_running:
            # Start monitoring with the current settings
            interval = st.session_state.monitoring_options['update_interval']
            initialize_monitoring(update_interval=interval)
        else:
            stop_monitoring()

def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.title("Configuration")
    
    # Update interval slider
    interval = st.sidebar.slider(
        "Update Interval (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state.monitoring_options['update_interval'],
        step=0.05,
        help="Time between updates. Lower values give more real-time updates but use more CPU."
    )
    
    # History size slider
    history_size = st.sidebar.slider(
        "History Size",
        min_value=10,
        max_value=100,
        value=st.session_state.monitoring_options['history_size'],
        step=10,
        help="Number of historical data points to keep for trends."
    )
    
    # Update options if changed
    if (interval != st.session_state.monitoring_options['update_interval'] or 
        history_size != st.session_state.monitoring_options['history_size']):
        with state_lock:
            # Store the new settings
            st.session_state.monitoring_options['update_interval'] = interval
            st.session_state.monitoring_options['history_size'] = history_size
            
            # Restart monitoring if it's running
            if st.session_state.monitoring_running:
                stop_monitoring()
                initialize_monitoring(update_interval=interval)
                st.sidebar.success("✅ Settings applied")

def main():
    """Main application entry point."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Real-time System Monitor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header and description
    st.title("Real-time System Monitor")
    st.caption("A comprehensive system monitoring dashboard")
    
    # Render sidebar with options
    render_sidebar()
    
    # Monitoring toggle switch
    toggle = st.toggle("Monitoring On/Off", value=st.session_state.monitoring_running)
    if toggle != st.session_state.monitoring_running:
        toggle_monitoring()
    
    # Create main display area
    display_area = st.empty()
    
    # Main monitoring loop - uses while loop with placeholder for real-time updates
    if st.session_state.monitoring_running:
        initialize_monitoring(update_interval=st.session_state.monitoring_options['update_interval'])
        while toggle:
            try:
                # Get the latest snapshot and render it
                snapshot = get_monitor().get_snapshot()
                draw_monitoring_dashboard(snapshot, display_area)
                
                # Sleep briefly to avoid hogging the CPU while maintaining real-time updates
                time.sleep(UPDATE_INTERVAL)
                
                # Check if toggle state changed
                if not st.session_state.monitoring_running:
                    break
                    
            except Exception as e:
                display_area.error(f"Error updating dashboard: {str(e)}")
                time.sleep(1)  # Wait before retrying on error
                
        # Ensure monitoring is stopped when loop exits
        stop_monitoring()
    else:
        # Show an informative message when monitoring is off
        with display_area.container():
            st.info("Monitoring is turned off. Toggle the switch above to start monitoring.")
            st.caption("Adjust update interval and history size in the sidebar.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log any unexpected errors
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Always ensure monitoring is stopped when app exits
        stop_monitoring()