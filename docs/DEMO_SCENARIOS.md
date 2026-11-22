# Demo scenario plan

Run each scenario with real, legally usable sample inputs. Do not label a scenario passed until its output has been manually reviewed.

1. Hindi WAV voice note → Bodhan transcription → English SRT.
2. Scanned multi-page chapter PDF → Bodhan OCR → English summary → Bodhan Tamil translation → Tamil WAV summary.
3. Worksheet photo → Bodhan OCR with Markdown table output → Bodhan Marathi translation.
4. Five short audio clips → transcribe with Bodhan → classify Telugu transcripts → translate only Telugu clips to English.
5. English paragraph → Bodhan Bengali translation → female-voice Bengali WAV → Bodhan back-transcription verification.
6. MP4/video request requiring French output → honest unsupported-language response; no fake translated artifact.

For every run, preserve `result.md`, `report.json`, `trace.jsonl` and generated artifacts. Capture at least one successful MCP tool call in an off-the-shelf MCP client separately from the CLI demonstrations.
