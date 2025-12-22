#!/bin/bash
# Training monitor - checks every 30 minutes

LOG_FILE="/tmp/claude/-home-zekiya-liquidity-uniswap-v3-simulator/tasks/bbedfe6.output"
MONITOR_LOG="/tmp/training_monitor.log"

echo "=== Training Monitor Started ===" > $MONITOR_LOG
echo "Checking every 30 minutes" >> $MONITOR_LOG
echo "" >> $MONITOR_LOG

while true; do
    # Check if training process is running
    if ! pgrep -f "train_ppo_v2.py" > /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Training COMPLETED" >> $MONITOR_LOG
        break
    fi

    # Get latest stats
    STEPS=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "total_timesteps" | tail -1 | grep -oP '\d+')
    REWARD=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "ep_rew_mean" | tail -1 | grep -oP '[\d.]+')
    FPS=$(tail -50 "$LOG_FILE" 2>/dev/null | grep "|    fps" | tail -1 | grep -oP '\d+')

    if [ -n "$STEPS" ]; then
        PROGRESS=$(echo "scale=1; $STEPS * 100 / 5000000" | bc)
        REMAINING=$((5000000 - STEPS))
        if [ -n "$FPS" ] && [ "$FPS" -gt 0 ]; then
            ETA_SEC=$((REMAINING / FPS))
            ETA_MIN=$((ETA_SEC / 60))
            ETA_HR=$((ETA_MIN / 60))
            ETA_MIN=$((ETA_MIN % 60))
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Steps: $STEPS (${PROGRESS}%) | Reward: $REWARD | ETA: ${ETA_HR}h ${ETA_MIN}m" >> $MONITOR_LOG
    fi

    # Wait 30 minutes
    sleep 1800
done
