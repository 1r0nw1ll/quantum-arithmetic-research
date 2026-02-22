# AG2 Guardrails POC

Minimal runnable demo of:

- `read_file` succeeds
- injected context attempts `send_webhook`
- middleware blocks escalation with deterministic failure artifacts

## Run

```bash
cd ag2_guardrails_poc
python main.py
```

## Run With AG2 (Live Tool Calls)

```bash
cd ag2_guardrails_poc
pip install "ag2[openai]"
python ag2_live_runner.py
```

AG2 config options:

- Put AG2/OpenAI config JSON at `ag2_guardrails_poc/OAI_CONFIG_LIST`, or
- Start from `ag2_guardrails_poc/OAI_CONFIG_LIST.example`, or
- Copy `ag2_guardrails_poc/.env.example` to `ag2_guardrails_poc/.env`, or
- Set `AG2_CONFIG_PATH=/path/to/config.json`, or
- Set `OPENAI_API_KEY` and optionally `AG2_MODEL` (default: `gpt-4o-mini`)

## Expected artifacts

- `out/artifacts/tool_call_step000_attempt.json`
- `out/artifacts/tool_call_step000_ok.json`
- `out/artifacts/tool_call_step001_attempt.json`
- `out/artifacts/prompt_injection_obstruction_step001.json`
- `out/artifacts/tool_call_step001_fail.json`
- `out/trace.jsonl`
