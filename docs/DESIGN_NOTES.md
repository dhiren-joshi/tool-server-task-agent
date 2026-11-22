# Design notes

## Tool granularity

The server exposes one public tool per primary user operation rather than one generic
"call model" tool. Each tool has a narrow schema that tells a third-party MCP client which
file type, language, output path and optional settings it needs. Helper tools (`inspect_files`,
`read_text_file`, and `write_english_text`) keep filesystem inspection and English-only planning
separate from Bodhan model operations. The agent and MCP server call the same Python tool layer,
so model limits and validation are not duplicated in two implementations.

## Media flow and context control

Media is referenced by local file paths. The planner receives an input inventory containing
file metadata, not raw audio or image bytes. OCR renders scanned PDF pages locally and sends
one page per Bodhan request; image bytes are encoded only in the outbound Bodhan request.
Audio decoding, chunking, concatenation and optional WAV-to-MP3 conversion happen in local
Python/FFmpeg code. They are not delegated to the LLM.

## Bodhan model boundaries

- Indic-Transcribe is the only speech-to-text provider. Audio is converted to WAV and split
  into 29-second windows.
- Indic-Translate is the only translation provider. Long text is chunked, empty or truncated
  responses are rejected, and back-translation is used when the source language is known
  reliably.
- Indic-Speak is the only speech generator. Text is sent in short chunks; generated audio is
  joined locally and checked by Bodhan Indic-Transcribe. The output is labelled AI-generated.
- Indic-OCR is the only OCR provider. Multi-page PDFs are rendered and sent page by page.

No Whisper, Tesseract, third-party translation, cloud speech, or commercial TTS fallback is
used. FFmpeg and PyMuPDF are media plumbing, not AI language models.

## Agent loop and untrusted content

The LLM chooses the next tool and writes English explanations. Indian-language output must be
returned by Bodhan Translate or Bodhan Speak. Text extracted from user files, OCR, transcripts
and filenames is explicitly marked as untrusted data in the system prompt; it cannot override
the user's task or the agent's model-use rules. The agent uses one tool call at a time so each
result can inform the next plan.

## Verification

Translation responses are checked for empty/truncated chunks. Where source language can be
determined reliably, the result is back-translated with Bodhan and compared as a diagnostic;
script ambiguity is not guessed. SRT generation verifies that every non-empty segment received
a translation, but its timestamps are interpolated inside chunk boundaries rather than aligned
word by word. Generated speech is back-transcribed with Bodhan and receives a similarity
diagnostic plus an AI-generated label. These checks are useful signals, not human quality review.

## Budgets, retries and caching

The task agent checks an estimated Bodhan cost before starting a tool and checks returned
usage after completion. It stops before later calls when the remaining budget is insufficient.
The estimate for translation and speech verification is not a mathematical guarantee of the
provider's final bill: actual output usage can differ from an estimate, so the report records
any post-call overrun and stops subsequent Bodhan requests. LLM provider charges are not included
in `total_bodhan_cost_inr`.

Transient Bodhan network/server errors and rate limits use bounded retry/backoff. Credit/balance
errors are not retried blindly. A per-run cache avoids repeating the same paid request; the
speech cache key includes the requested output format so WAV and MP3 artifacts are never
confused.

## Filesystem safety

The MCP server confines inputs and outputs to its configured workspace. It refuses to write
`.env`-style files, `.git`, `.venv`, Python cache folders, or overwrite existing workspace files
outside the dedicated `outputs/` directory. CLI output names are simple filenames written to
the run's `artifacts/` directory.

## Known limitations

- API integrations and language quality still need live validation with Bodhan credentials.
- A source language that cannot be identified confidently from script alone requires the agent
  to supply `source_language` for back-translation; otherwise the report says `primary_only`.
- SRT timing is approximate segment/sentence timing, not word-level forced alignment.
- The rupee budget is an estimate-first guard with post-call reconciliation; see the budget note
  above for the limitation.
- The current offline test suite uses fake provider responses. It cannot establish live model
  quality, real provider latency, actual billed cost, or voice quality.
