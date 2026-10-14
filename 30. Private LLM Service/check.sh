#!/usr/bin/env bash
# Checks the private LLM service over the network, from any machine:
# access, auth, one chat answer, max context, N parallel requests, rate limit.
# Needs only curl and python3.
#
# Usage: bash check.sh http://<vds-host>:4030 <api-key> [parallel requests, default 8]
set -uo pipefail

BASE="${1:?usage: bash check.sh http://host:port <api-key> [parallel]}"
BASE="${BASE%/}"
KEY="${2:?usage: bash check.sh http://host:port <api-key> [parallel]}"
N="${3:-8}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }

# post <file-for-body> [key] → prints "<status> <seconds>", body in $TMP/last
post() {
  local auth=()
  [ -n "${2:-}" ] && auth=(-H "Authorization: Bearer $2")
  curl -s -o "$TMP/last" -w '%{http_code} %{time_total}' ${auth[@]+"${auth[@]}"} \
    -H 'Content-Type: application/json' --data-binary "@$1" "$BASE/v1/chat/completions"
}

echo "== 1. Network access: $BASE/health (public)"
curl -s -m 10 "$BASE/health" | json "f\"service {d['status']}, ollama reachable={d['ollama']['reachable']} {d['ollama'].get('version') or d['ollama'].get('error')}\"" \
  || { echo "service unreachable"; exit 1; }

echo
echo "== 2. Auth: no key, wrong key"
echo '{"messages":[{"role":"user","content":"hi"}]}' > "$TMP/hi.json"
read -r code _ < <(post "$TMP/hi.json")
echo "no key    → $code $(json "d['error']['message']" < "$TMP/last")"
read -r code _ < <(post "$TMP/hi.json" "sk-local-guess")
echo "wrong key → $code $(json "d['error']['message']" < "$TMP/last")"

echo
echo "== 3. One chat request"
echo '{"messages":[{"role":"user","content":"Какая столица Франции? Ответь одним словом."}],"temperature":0}' > "$TMP/one.json"
read -r code secs < <(post "$TMP/one.json" "$KEY")
if [ "$code" = 200 ]; then
  echo "$code in ${secs}s → $(json "repr(d['choices'][0]['message']['content'])" < "$TMP/last")"
  json "f\"usage {d['usage']}, {d['service']['tokens_per_sec']:.1f} tok/s, {d['service']['rate_limit_remaining']} req left this minute\"" < "$TMP/last"
else
  echo "$code $(json "d['error']['message']" < "$TMP/last")"
fi

echo
echo "== 4. Max context: a ~26 KB prompt"
python3 -c "import json; print(json.dumps({'messages':[{'role':'user','content':'context test '*2000}]}))" > "$TMP/long.json"
read -r code secs < <(post "$TMP/long.json" "$KEY")
echo "$code in ${secs}s → $(json "d['error']['message']" < "$TMP/last")"

echo
echo "== 5. Stability: $N requests at once"
start=$(python3 -c 'import time; print(time.time())')
for i in $(seq 1 "$N"); do
  printf '{"messages":[{"role":"user","content":"Request %s: name one random animal, one word."}],"max_tokens":16}' "$i" > "$TMP/p$i.json"
  ( curl -s -o "$TMP/r$i" -w '%{http_code} %{time_total}' -H "Authorization: Bearer $KEY" \
      -H 'Content-Type: application/json' --data-binary "@$TMP/p$i.json" "$BASE/v1/chat/completions" > "$TMP/s$i" ) &
done
wait
for i in $(seq 1 "$N"); do
  read -r code secs < "$TMP/s$i"
  if [ "$code" = 200 ]; then
    detail=$(json "f\"{d['choices'][0]['message']['content'].strip()[:30]!r} queue {d['service']['queue_ms']} ms\"" < "$TMP/r$i")
  else
    detail=$(json "d['error']['code']" < "$TMP/r$i" 2>/dev/null || echo "no body")
  fi
  printf '#%-2s %s %6.2fs  %s\n' "$i" "$code" "$secs" "$detail"
done
echo "summary: $(awk '{print $1}' "$TMP"/s[0-9]* | sort | uniq -c | awk '{printf "%s × %s  ", $1, $2}')in $(python3 -c "import time; print(f'{time.time()-$start:.1f}')") s"

echo
echo "== 6. Rate limit: invalid requests (cheap, never reach the model) until 429"
echo '{"messages":[]}' > "$TMP/empty.json"
for i in $(seq 1 200); do
  read -r code _ < <(post "$TMP/empty.json" "$KEY")
  if [ "$code" = 429 ]; then
    retry=$(curl -s -D - -o /dev/null -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
      --data-binary "@$TMP/empty.json" "$BASE/v1/chat/completions" | awk 'tolower($1)=="retry-after:"{print $2}' | tr -d '\r')
    echo "request #$i → 429 $(json "d['error']['message']" < "$TMP/last") (Retry-After: $retry s)"
    break
  fi
done

echo
echo "== Service stats"
curl -s -H "Authorization: Bearer $KEY" "$BASE/api/stats" \
  | json "f\"by status {d['by_status']}, latency {d['latency_ms']}, running {d['running']}, queued {d['queued']}\""
