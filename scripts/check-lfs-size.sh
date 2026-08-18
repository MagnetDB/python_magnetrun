#!/usr/bin/env bash
ORG="MagnetDB"

C1=40  # depot path column width
C2=10  # float width for LFS size column

_sep() { printf '─%.0s' $(seq 1 "$1"); }
_top() { printf "┌%s┬%s┐\n" "$(_sep $((C1+2)))" "$(_sep $((C2+5)))"; }
_mid() { printf "├%s┼%s┤\n" "$(_sep $((C1+2)))" "$(_sep $((C2+5)))"; }
_bot() { printf "└%s┴%s┘\n" "$(_sep $((C1+2)))" "$(_sep $((C2+5)))"; }

printf '\n'
_top
printf "│ \033[1m%-${C1}s\033[0m │ \033[1m%$((C2+3))s \033[0m│\n" "Depot Path" "LFS Size"
_mid

data=$(for depot in ../*/; do
  if [ -d "$depot/.git" ]; then
    remote_url=$(cd "$depot" && git config --get remote.origin.url 2>/dev/null)
    if [[ "$remote_url" == *"$ORG"* ]]; then
      bytes=$(cd "$depot" && git lfs ls-files --debug 2>/dev/null \
              | awk -F: '$1~/size/{t+=$2} END {print t+0}')
      if [ "$bytes" -gt 0 ]; then
        mb=$(awk "BEGIN {print $bytes/1024/1024}")
        printf "│ %-${C1}s │ %${C2}.2f MB │\n" "$depot" "$mb"
      fi
    fi
  fi
done)

echo "$data"
_mid

total_mb=$(echo "$data" | awk '{sum+=$4} END {print sum+0}')
printf "│ \033[1m%-${C1}s\033[0m │ \033[1m%${C2}.2f MB \033[0m│\n" "TOTAL" "$total_mb"
_bot
printf '\n'
