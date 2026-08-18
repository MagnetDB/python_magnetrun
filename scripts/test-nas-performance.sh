#! /bin/bash

NAS_AUTOFS=/mnt/LNCMIG-Data/records
NAS_RCLONE=/home/LNCMI-G/christophe.trophime/LNCMIG-Data/records

# Find the largest readable .tdms in NAS_RCLONE (scan largest-first, stop at first readable)
RCLONE_LARGE_FILE=$(find "${NAS_RCLONE}" -name "*.tdms" -printf "%s\t%p\n" 2>/dev/null | sort -rn | while IFS="$(printf '\t')" read -r size path; do
    if head -c1 "$path" >/dev/null 2>&1; then
        echo "$path"
        break
    fi
done)
AUTOFS_LARGE_FILE="${NAS_AUTOFS}${RCLONE_LARGE_FILE#"${NAS_RCLONE}"}"

# Trigger autofs mount before any test
ls "${NAS_AUTOFS}" >/dev/null 2>&1

echo "Large read target (rclone): ${RCLONE_LARGE_FILE}"
echo "Large read target (autofs): ${AUTOFS_LARGE_FILE}"

run_fio() {
    local name="$1" file="$2" rw="$3" bs="$4" size="$5" runtime="$6" time_based="$7"
    if [ -f "$file" ]; then
        fio --name="$name" --filename="$file" --rw="$rw" --bs="$bs" --size="$size" \
            --numjobs=1 --runtime="$runtime" --time_based="$time_based" --readonly
    else
        echo "WARNING: ${file} does not exist — skipping fio test ${name}"
    fi
}

# === TEST 1: SSHFS LARGE READ ===
sync && echo "=== autofs LARGE READ ===" && echo 3 | sudo tee /proc/sys/vm/drop_caches
run_fio sshfs_seq_read "${AUTOFS_LARGE_FILE}" read 1M 1G 60 0

# === TEST 2: RCLONE LARGE READ ===
sync && echo "=== rclone LARGE READ ===" && echo 3 | sudo tee /proc/sys/vm/drop_caches
run_fio rclone_seq_read "${RCLONE_LARGE_FILE}" read 1M 1G 60 0

# === TEST 3: SSHFS SMALL RANDOM READ ===
sync && echo "=== autofs SMALL RANDOM READ ===" && echo 3 | sudo tee /proc/sys/vm/drop_caches
run_fio sshfs_rand_read "${AUTOFS_LARGE_FILE}" randread 4k 100M 30 1

# === TEST 4: RCLONE SMALL RANDOM READ ===
sync && echo "=== rclone SMALL RANDOM READ ===" && echo 3 | sudo tee /proc/sys/vm/drop_caches
run_fio rclone_rand_read "${RCLONE_LARGE_FILE}" randread 4k 100M 30 1

# === TEST 5: SSHFS METADATA SPEED ===
sync  && echo "=== autofs METADATA SPEED ===" && echo 3 | sudo tee /proc/sys/vm/drop_caches && time find ${NAS_AUTOFS}/pbsurv -type f | wc -l

# === TEST 6: RCLONE METADATA SPEED ===
sync && echo "=== rclone METADATA SPEED ===" && echo 3 | sudo tee /proc/sys/vm/drop_caches && time find ${NAS_RCLONE}/pbsurv -type f | wc -l
