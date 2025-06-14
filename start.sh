#!/bin/bash

# Function to handle errors
handle_error() {
    local line_no=$1
    local command=$2
    local command=$3
    local exit_code=$4
    echo "❌ ERROR at line ${line_no}: Command '${command}' exited with status ${exit_code}"
    echo ""
}

# Set up error trap to catch and report errors
trap 'handle_error ${LINENO} "$BASH_COMMAND" $?' ERR

# Function to display usage instructions
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --all       Launch frontend, backend, and monitoring (default if no option is specified)"
    echo "  --backend   Launch only the backend"
    echo "  --frontend  Launch only the frontend"
    echo "  --monitoring   Launch only the hardware monitoring"
    echo "  --help      Display this help message"
    echo ""
    exit 1
}

# Activate virtual environment
activate_venv() {
    echo "📜 Activating virtual environment..."
    if [ ! -d ".venv" ]; then
        echo "❌ ERROR: Virtual environment .venv not found!"
        echo "💬 Please create the virtual environment first."
        echo ""
        return 1
    fi
    
    if [ ! -f ".venv/bin/activate" ]; then
        echo "❌ ERROR: Virtual environment appears damaged (activate script not found)."
        echo ""
        return 1
    fi
    
    source .venv/bin/activate || {
        echo "❌ ERROR: Failed to activate virtual environment!"
        echo ""
        return 1
    }
    
    echo "✅ Virtual environment activated successfully."
    echo ""
    return 0
}

# Check if backend is responsive
check_backend_health() {
    echo "💬 Checking backend health..."
    # Wait for the backend to start (adjust time as needed)
    sleep 60
    
    # Use curl to check the backend health
    if command -v curl &> /dev/null; then
        echo "Pinging backend at http://localhost:8000/ping"
        RESPONSE=$(curl -s http://localhost:8000/ping) || {
            echo "⚠️ WARNING: Failed to connect to backend."
            echo "💬 Backend may not be running correctly."
            echo ""
            return 1
        }
        echo "💬 Backend response: $RESPONSE"
        echo ""
    else
        # Fallback to using wget if curl is not available
        if command -v wget &> /dev/null; then
            echo "Pinging backend at http://localhost:8000/ping"
            RESPONSE=$(wget -qO- http://localhost:8000/ping) || {
                echo "⚠️ WARNING: Failed to connect to backend."
                echo "💬 Backend may not be running correctly."
                echo ""
                return 1
            }
            echo "💬 Backend response: $RESPONSE"
            echo ""
        else
            echo "⚠️ WARNING: Neither curl nor wget is installed. Cannot check backend health."
            echo ""
            return 1
        fi
    fi
    
    return 0
}


# Launch monitoring
launch_monitoring() {
    echo "🚀 Launching monitoring..."
    
    # Check if streamlit exists in venv
    if [ ! -f ".venv/bin/streamlit" ]; then
        echo "❌ ERROR: Streamlit not found in virtual environment."
        echo "💬 Try installing it with: pip install streamlit"
        echo ""
        return 1
    fi
    
    # Check if frontend file exists
    if [ ! -f "system_monitoring/main.py" ]; then
        echo "❌ ERROR: System Monitoring file 'system_monitoring/main.py' not found!"
        echo ""
        return 1
    fi
    
    # Run streamlit
    .venv/bin/streamlit run system_monitoring/main.py --server.port 8502 &
    MONITORING_PID=$!
    echo "✅ Hardware monitoring started with PID: $MONITORING_PID"
    echo ""
    
    # Add PID to the process group this script controls
    PIDS+=($MONITORING_PID)
    return 0
}

# Launch backend
launch_backend() {
    echo "🚀 Launching backend..."
    
    # Check if python3.10 exists
    if ! command -v python3.10 &> /dev/null; then
        echo "❌ ERROR: python3.10 not found. Please install it or modify the script to use your Python version."
        echo ""
        return 1
    fi
    
    # Check if backend file exists
    if [ ! -f "backend/main.py" ]; then
        echo "❌ ERROR: Backend file 'backend/main.py' not found!"
        echo ""
        return 1
    fi
    
    python3.10 backend/main.py &
    BACKEND_PID=$!
    echo "✅ Backend started with PID: $BACKEND_PID"
    echo ""

    # Add PID to the process group this script controls
    PIDS+=($BACKEND_PID)
    
    ## Check if backend is responsive
    #check_backend_health
    return $?
}

# Launch frontend
launch_frontend() {
    echo "🚀 Launching frontend..."
    
    # Check if streamlit exists in venv
    if [ ! -f ".venv/bin/streamlit" ]; then
        echo "❌ ERROR: Streamlit not found in virtual environment."
        echo "💬 Try installing it with: pip install streamlit"
        echo ""
        return 1
    fi
    
    # Check if frontend file exists
    if [ ! -f "frontend/main.py" ]; then
        echo "❌ ERROR: Frontend file 'frontend/main.py' not found!"
        echo ""
        return 1
    fi
    
    # Run streamlit
    .venv/bin/streamlit run frontend/main.py --server.port 8501 &
    FRONTEND_PID=$!
    echo "✅ Frontend started with PID: $FRONTEND_PID"
    echo ""
    
    # Add PID to the process group this script controls
    PIDS+=($FRONTEND_PID)
    return 0
}

# Cleanup function
cleanup() {
    echo ""
    echo "ℹ️  Shutting down processes..."
    
    # Kill all processes in our process group
    for PID in "${PIDS[@]}"; do
        if ps -p $PID > /dev/null; then
            echo "💬 Stopping process with PID: $PID"
            kill $PID 2>/dev/null || true
            # Wait briefly to see if it terminates gracefully
            sleep 0.5
            # Force kill if still running
            if ps -p $PID > /dev/null; then
                echo "💬 Process $PID still running, forcing termination..."
                kill -9 $PID 2>/dev/null || true
            fi
        fi
    done
    
    echo "ℹ️  All processes stopped"
    echo ""
    exit 0
}

# Set trap to catch signals
trap cleanup SIGINT SIGTERM EXIT

# Array to store PIDs of all launched processes
PIDS=()

# Default behavior - launch all
LAUNCH_BACKEND=false
LAUNCH_FRONTEND=false
LAUNCH_MONITORING=false

# If no arguments, launch all
if [ $# -eq 0 ]; then
    LAUNCH_BACKEND=true
    LAUNCH_FRONTEND=true
    LAUNCH_MONITORING=true
else
    # Parse command line arguments
    for arg in "$@"
    do
        case $arg in
            --all)
                LAUNCH_BACKEND=true
                LAUNCH_FRONTEND=true
                LAUNCH_MONITORING=true
                ;;
            --backend)
                LAUNCH_BACKEND=true
                ;;
            --frontend)
                LAUNCH_FRONTEND=true
                ;;
            --monitoring)
                LAUNCH_MONITORING=true
                ;;
            --help)
                usage
                ;;
            *)
                echo "Unknown option: $arg"
                usage
                ;;
        esac
    done
fi

# Activate the virtual environment first
if ! activate_venv; then
    echo "❌ Failed to activate virtual environment. Exiting."
    echo ""
    exit 1
fi

# Launch components according to parameters
if $LAUNCH_BACKEND; then
    if ! launch_backend; then
        echo "❌ Failed to launch backend. Continuing with remaining tasks..."
        echo ""
    fi
fi

if $LAUNCH_FRONTEND; then
    if ! launch_frontend; then
        echo "❌ Failed to launch frontend. Continuing with remaining tasks..."
        echo ""
    fi
fi

if $LAUNCH_MONITORING; then
    if ! launch_monitoring; then
        echo "❌ Failed to launch monitoring. Continuing with remaining tasks..."
        echo ""
    fi
fi

# Report what's running
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "ℹ️  Running processes: ${PIDS[*]}"
    echo ""
    echo "✅ Services are running. Press Ctrl+C to stop all processes..."
    echo ""
    
    # Keep the script running until all child processes exit
    # or until we receive a SIGINT or SIGTERM
    while true; do
        # Check if any process is still running
        all_dead=true
        for PID in "${PIDS[@]}"; do
            if ps -p $PID > /dev/null; then
                all_dead=false
                break
            fi
        done
        
        # If all processes have died, exit
        if $all_dead; then
            echo "✅ All processes have exited."
            echo ""
            exit 0
        fi
        
        # Sleep briefly before checking again
        sleep 1
    done
else
    echo "⚠️ No services were successfully started."
    echo ""
    exit 1
fi