#!/bin/bash
set -euo pipefail

LOG_FILE="logs/health.log"
mkdir -p logs

disk_free=$(df -h . | awk 'NR==2 {print $4}')
mem_free=$(vm_stat | awk 'NR==3 {print $3}' | sed 's/\.//')
cpu_load=$(uptime | awk -F'load average:' '{print $2}')

echo "[$(date '+%F %T')] Disk: $disk_free | Free RAM: ${mem_free}MB | Load: $cpu_load" | tee -a "$LOG_FILE"

if [ "$(df . | awk 'NR==2 {print $4}')" -lt 2000000 ]; then
  echo "[ERROR] Low disk space!" | tee -a "$LOG_FILE"
  exit 1
fi
