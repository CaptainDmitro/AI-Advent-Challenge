#!/usr/bin/env bash
# The lesson without Rust: the same local model through Ollama's CLI and both of
# its HTTP APIs. Needs `ollama serve` running, `curl` and `python3`.
set -euo pipefail

HOST="${OLLAMA_HOST:-http://localhost:11434}"
[[ "$HOST" == *://* ]] || HOST="http://$HOST"
MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"

echo "== Server and models =="
curl -sf "$HOST/api/version"; echo
ollama list

echo
echo "== 1. easy · CLI: ollama run =="
ollama run "$MODEL" "Какая столица Франции? Ответь одним словом."

echo
echo "== 2. medium · HTTP: native /api/chat =="
curl -sf "$HOST/api/chat" -d @- <<EOF | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(r["message"]["content"])
print("[%d -> %d tok, %.1f tok/s, total %.2fs, model load %.2fs]" % (
    r["prompt_eval_count"], r["eval_count"], r["eval_count"] / r["eval_duration"] * 1e9,
    r["total_duration"] / 1e9, r["load_duration"] / 1e9))'
{"model": "$MODEL", "stream": false, "options": {"temperature": 0},
 "messages": [{"role": "user", "content": "Объясни в трёх предложениях, чем TCP отличается от UDP, и назови по одному протоколу или сервису, который работает поверх каждого из них."}]}
EOF

echo
echo "== 3. hard · HTTP: OpenAI-compatible /v1/chat/completions =="
curl -sf "$HOST/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<EOF | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(r["choices"][0]["message"]["content"])
u = r["usage"]
print("[%d -> %d tok] expected answer: 11:36" % (u["prompt_tokens"], u["completion_tokens"]))'
{"model": "$MODEL", "temperature": 0,
 "messages": [{"role": "user", "content": "Поезд выехал из города A в 9:00 со скоростью 60 км/ч. Второй поезд выехал из города B ему навстречу в 10:00 со скоростью 90 км/ч. Расстояние между городами 300 км. Во сколько они встретятся? Реши по шагам, а в последней строке напиши только время в формате ЧЧ:ММ."}]}
EOF
