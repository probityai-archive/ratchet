# If the network fails

Steps 1 to 6 of [run_demo.sh](run_demo.sh) read Atlas. Step 7 reads a local file and always works.
If a step hangs for more than 10 seconds, press Ctrl-C and switch to the recording below. Keep the spoken script
of [PRESENTATION.md](PRESENTATION.md) unchanged; say once, "this is a recording of the same commands from this
afternoon", and continue.

## The recording

[`docs/demo/out/20261011T232221Z/ratchet-169.mp4`](../out/20261011T232221Z/ratchet-169.mp4): 35.6 seconds,
1920x1080, no audio, captions burned in, banner "RECORDED RUN 2026-10-11 16:23 PT | real commands, unedited".
Every command exited 0 ([manifest](../out/20261011T232221Z/manifest-169.json)). Play it muted, pause on each
scene, and speak over it.

| Script section | Clip time | Scene (the same command as the live step) |
|---|---|---|
| How it works (step 1) | 0:00-0:07 | `ratchet show trace` - the double refund and its grader lines |
| Case one REJECTED, case two ADMITTED (step 2) | 0:07-0:16 | `ratchet replay` - the A/B table; pause on VERDICT |
| Case two ADMITTED (step 3) | 0:16-0:25 | `ratchet show decision` - the gate table, all PASS |
| Verify (step 4) | 0:25-0:29 | `ratchet verify` - VERIFIED |
| New tool refused (step 5) | 0:29-0:35 | `ratchet show decision` - authority FAIL |

If the video player fails too, the same take plays in the terminal: `asciinema play
docs/demo/out/20261011T232221Z/run-169.cast`. The [contact sheet](../out/20261011T232221Z/contact-169.png) shows
all five scenes on one image as a last resort.

## Steps 6 and 7 without the network

The recording does not contain steps 6 and 7. Both have local sources:

- Now a real model (step 6): `cat docs/results/v3-live.md`. Point at the table rows "33.4%" and "42.1%" and at
  "`policy_changed.v6` fell from 21 to 8 of 40". The script's words stay the same.
- Real data (step 7): `cat docs/results/hooks-replay.md`, exactly as on stage.
