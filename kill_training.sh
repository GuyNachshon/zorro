#!/bin/bash
# Kill hanging training processes

echo "🛑 Killing hanging training processes..."

# Kill the specific hanging processes
pkill -f "train_icn.py"
pkill -f "train_all.py"

# Wait a moment
sleep 2

# Check if processes are killed
if pgrep -f "train_icn.py" > /dev/null; then
    echo "⚠️ Some processes still running, using stronger signal..."
    pkill -9 -f "train_icn.py"
    pkill -9 -f "train_all.py"
fi

echo "✅ Training processes killed"

# Show remaining python processes
echo ""
echo "Remaining Python processes:"
ps aux | grep python | grep -v grep

echo ""
echo "You can now run the fixed training script:"
echo "  python icn_training_fix.py"
echo ""
echo "Or for a complete fix, run:"
echo "  python train_all_fixed.py"