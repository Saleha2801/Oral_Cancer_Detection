#!/bin/bash

# ==============================================================================
# Oral Cancer AI - Full Stack Launcher
# Starts both FastAPI Backend (port 8000) and Vite React Frontend (port 5173)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================="
echo "   🧬 Oral Cancer AI: Multimodal Detection System"
echo "========================================================="

# 1. Check Python virtual environment
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment python not found at $VENV_PYTHON"
    exit 1
fi

# 2. Check model weights
echo "Checking trained model checkpoints in models/ ..."
for m in "best_oral_cancer_resnet18.pth" "best_histopathology_model.pth" "clinical_model.joblib"; do
    if [ -f "$SCRIPT_DIR/models/$m" ] || [ -f "$SCRIPT_DIR/$m" ]; then
        echo "   ✅ Found: $m"
    else
        echo "   ⚠️ Missing: $m"
    fi
done

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    echo "Oral Cancer AI services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 3. Start Backend
echo ""
echo "🚀 Starting FastAPI Backend on http://127.0.0.1:8000 ..."
"$SCRIPT_DIR/.venv/bin/uvicorn" backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait for backend to come online
echo "Waiting for backend..."
sleep 2

# 4. Start Frontend
echo ""
echo "🚀 Starting Vite React Frontend on http://localhost:5173 ..."
cd "$SCRIPT_DIR/frontend"
npm run dev -- --host 127.0.0.1 &
FRONTEND_PID=$!

echo ""
echo "========================================================="
echo "   ✨ Application is running!"
echo "   - Frontend: http://localhost:5173"
echo "   - Backend:  http://127.0.0.1:8000"
echo "   - API Docs: http://127.0.0.1:8000/docs"
echo "   Press Ctrl+C to stop all services."
echo "========================================================="

# Wait indefinitely
wait
