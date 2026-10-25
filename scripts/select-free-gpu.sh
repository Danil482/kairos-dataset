#!/usr/bin/env bash
# Metadata-only selector. This does not reserve, mount or stop any GPU/process.
set -euo pipefail
minimum_free=${1:-18000}
preferred=${2:-}
[[ "$minimum_free" =~ ^[0-9]+$ ]]
inventory=$(nvidia-smi --query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits) || exit 69
choice=
while IFS=, read -r index uuid name free used utilization; do
  index=${index//[[:space:]]/}; uuid=${uuid//[[:space:]]/}
  free=${free//[[:space:]]/}; used=${used//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  name=${name# }
  [[ "$index" =~ ^[0-9]+$ && "$uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]] || continue
  [[ "$free" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || continue
  [[ "$free" -ge "$minimum_free" && "$used" -le 64 && "$utilization" -le 5 ]] || continue
  processes=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits) || continue
  [[ -z ${processes//[[:space:]]/} ]] || continue
  if [[ -z "$choice" || "$uuid" == "$preferred" ]]; then
    printf -v choice '%s\t%s\t%s\t%s' "$uuid" "$index" "$name" "$free"
  fi
  if [[ "$uuid" == "$preferred" ]]; then break; fi
done <<< "$inventory"
if [[ -z "$choice" ]]; then
  printf 'No idle GPU with at least %s MiB free; no processes changed.\n' "$minimum_free" >&2
  exit 75
fi
printf '%s\n' "$choice"
