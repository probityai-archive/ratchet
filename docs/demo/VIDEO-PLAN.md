# Ratchet: the 1-minute submission video

The video is a recording of real commands against the Atlas store, with a banner that says so and captions
that carry the claims. Recording at 3:45 PM PT is one command:

```sh
docs/demo/video/record.sh
```

It records, renders and checks both cuts, and refuses to finish if any gate fails. Output lands in
`docs/demo/out/<UTC stamp>/` (git-ignored).

## What it produces

| File | Use |
|---|---|
| `ratchet-169.mp4` | Submission form and anywhere a laptop plays it. 1920x1080, h264, yuv420p, 30 fps, faststart, silent unless narrated |
| `ratchet-45.mp4` | LinkedIn. 1080x1350 (4:5), a native narrow-terminal cut with 30 px type, never a padded 16:9 and never 9:16 |
| `ratchet-*.vtt` | The captions as WebVTT, for upload alongside either video |
| `contact-*.png` | One frame per step. Read it before using the video |
| `manifest-*.json` | Step commands and exit codes, caption text, tool digests, sha256 of the cast and every output |
| `run-*.cast` | The raw terminal recording, the primary source for every pixel in the video |

## What is real

- Every command in `steps.submission.toml` is executed by `drive.py` inside the recording terminal.
  The driver only paces the typing so a viewer can read it.
- The terminal pixels come from the cast alone. The banner reads `RECORDED RUN <time> PT | real commands,
  unedited` (the time is the recording's own start time). Evaluations and decisions on screen were
  written to Atlas earlier; their own timestamps appear in the output.
- Captions are text on the padding around the terminal. Numbers in captions are `{{PLACEHOLDERS}}`
  derived by `docs/demo/derive_ids.py` from Atlas and the stored results, with the source of each written
  beside it in `ids.env`. A missing value stops the recording before it starts.
- A step that exits with an unexpected code, or times out, fails the check. Nothing is retried quietly.
- No speed-up, cut or splice. If a take is too long, shorten a `hold` and record again.

## Time plan

The structure is the approved one-minute outline.

| Seconds | Step id | Command | On screen | Caption claim |
|---|---|---|---|---|
| 0-8 | `failure` | `ratchet show trace $BAD_TRACE` | commit, timeout, retry, two refunds in the ledger | Passing held-out tests is necessary; it does not show what a patch would block |
| 8-23 | `matrix` | `ratchet replay $REJECTED_CAND $CLEARED_CAND` | replay matrix, one guard falsely blocking a legitimate refund | replay count and time; the rejected guard |
| 23-38 | `admitted`, `authority` | `ratchet show decision $ADMITTED_DECISION`, then `$AUTHORITY_DECISION` | paired fresh results, benign twins, gates; authority growth rejected | parent and candidate passes of N; authority rule |
| 38-48 | `verify`, `lineage` | `ratchet verify ...`, `ratchet show lineage` | signature valid against the pinned key; Atlas lineage | signed receipt; Atlas record |
| 48-56 | `benchmarks`, `level2` | `cat $RESULTS_FILE`, `cat $LEVEL2_FILE` | benchmark subset sizes; the level-2 comparison | AgentDojo and tau-bench counts; level-2 result as measured |
| 56-60 | `level2` (second caption beat) | | last screen held | Failure -> executable change -> recorded checks -> signed admission |

Typing a command costs about 1.5 s and the step's own run time is real. `check.py` reads the measured
duration from the manifest and fails at more than 60.0 s. The first take after real data exists will
show which `hold` values to shorten; each step's `hold` is one number in the TOML.

## Pipeline

| Stage | Tool | Detail |
|---|---|---|
| Install | `video/setup.sh` | asciinema v3.2.1 and agg v1.9.0, downloaded from their GitHub releases and checked against pinned sha256 values, into `~/.local/share/ratchet-demo-tools` |
| Warm | `drive.py --warm` | runs every command once silently, so Python imports and the file cache are warm and the take is not dead air |
| Record | `asciinema rec --headless --window-size COLSxROWS` | a headless terminal of the exact size for the format: 123x26 for 16:9, 60x27 for 4:5 |
| Render | `render.py` | `agg` draws the terminal at 25 fps; `ffmpeg` pads it to the canvas, burns the banner and captions (Inter on the padding), and encodes h264 |
| Check | `check.py` | codec, pixel format, frame rate, canvas, faststart, size, duration at most 60 s, every step exit code, no unresolved placeholder, caption length, manifest digests, no test label |

The 4:5 cut uses a narrower terminal, so a step may carry a `cmd_45` (the matrix uses `--max-columns 2`).
Type is 30 px on a 1080 px canvas, which is the legibility floor for a phone feed.

### Why the recording is timed by the cast

The recorder stamps every output event with real time, so the cast is exact whatever the machine load,
and the renderer draws frames from those stamps. A recorder that grabs screenshots as fast as it can and
plays them back at a nominal rate shortens the video under load (measured on this workstation at load
average 18 to 31: a 4-second script became 0.84 s). A caption timed to a step needs the exact clock.

## Before 3:45 PM

1. `uv run python docs/demo/derive_ids.py` writes `docs/demo/ids.env` from Atlas and the stored results,
   or names each value that does not exist yet and stops. `record.sh` runs it first for a submission take.
2. `docs/demo/video/record.sh --test` (about 1 minute). It runs the same pipeline on placeholder commands
   and synthetic screens and stamps `PIPELINE TEST, NOT A SUBMISSION CAPTURE` on the banner. If the
   banner on a file does not say that, the file is a submission take.
3. Read `contact-169.png` and `contact-45.png` from the real take. Open the mp4 on a phone for the 4:5 cut.
4. Upload `ratchet-169.mp4` for the form; keep `ratchet-45.mp4` and its `.vtt` for LinkedIn.

## Narration (optional)

The cut is complete without sound: autoplay on LinkedIn is muted and the captions carry every claim.
To add a voice, record it against the finished video and run:

```sh
docs/demo/video/narrate.sh docs/demo/out/<stamp>/ratchet-169.mp4 narration.m4a
```

The video stream is copied, so pixels and caption timing do not change. The voice should be the
presenter's own. A draft to start from, about 125 words for 60 seconds, to be rewritten in the
presenter's wording and checked against the numbers on screen:

> A refund goes through, the connection times out, and the agent retries. Now the customer has two
> refunds. Passing held-out tests would not have told you what a patch would block.
> Ratchet replays each proposed guard over every recorded action. This one stops the retry, and also
> blocks a legitimate refund, so it is rejected. This one does not, so it goes to fresh evaluation.
> Parent against candidate, on cases the improver cannot read, with benign twins and an authority check.
> If it passes, the decision is signed, and anyone can verify the signature against a pinned key.
> Everything lives in Atlas: every version, every trace, every rejection. We ran it on AgentDojo and
> tau-bench retail, and the improver's own instructions go through the same test.
> Failure, executable change, recorded checks, signed admission.
