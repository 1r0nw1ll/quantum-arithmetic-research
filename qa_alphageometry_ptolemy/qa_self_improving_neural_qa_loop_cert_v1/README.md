# QA Self-Improving Neural QA Loop Cert v1

Registered family [525].

This cert validates multi-round loop transcripts for the replay-gated neural QA
learner. It proves that a bounded run can continue across multiple rounds while
preserving the rule that only accepted packets mutate promoted state.

The runner's candidate selection prefers bounded configuration proposals and
general-ML replay artifacts before legacy HSI replay artifacts, so broad globs
can remain enabled without forcing the learner to stay HSI-only.

## Files

| File | Purpose |
|---|---|
| `qa_self_improving_neural_qa_loop_cert_validate.py` | Cert-family validator. |
| `mapping_protocol_ref.json` | Gate 0 mapping reference. |
| `fixtures/pass_live_transcript.jsonl` | Live nine-round transcript. |
| `fixtures/fail_tampered_rejected_mutates.jsonl` | Negative fixture: rejected round claims promoted mutation. |

## Validation

```bash
python3 qa_alphageometry_ptolemy/qa_self_improving_neural_qa_loop_cert_v1/qa_self_improving_neural_qa_loop_cert_validate.py --self-test
```
