# Local implementation decisions

Updated 2026-09-20. These are two separate implementation units, following the accepted compact-UI iteration already in the working tree. No commit, platform release, or GPU launch has been made.

## Model table and configuration

Runtime paths: `src/ui/**`.

- Order: Bounding Box, Point-based, Smart Tool, Detector, Mask, Oriented Box.
- A local `ModelTable` widget renders purpose icons, selected model, independent status, and Configure/Change. Clicking a ready model name opens inference settings; there is no Settings column. It composes normal SDK widgets; no SDK changes are required.
- Styling uses the platform's `el-table` / `ultra-table`, as in SDK `RandomSplitsTable` and `MatchTagsOrClasses`. Standard SDK `Text` status icons/colors replace the local palette. The custom table stylesheet and its static-directory mount have been removed; the table uses a 5 px radius matching the native card.
- Configure opens one dialog containing existing sessions and the deployment form. **Deploy new** reveals the deployment form inline while keeping the session selector visible. There are no Existing/Deploy tabs. Empty catalogs show an explanation and the deploy action; interpolation appears in the same selector only for supported geometries. URL configuration follows the existing environment rule.
- Session/URL selection is a draft until **Use model**. Cancel/close retains the applied selection. Deployment itself starts a remote session and selects its exact task ID; closing the dialog does not undo a launched session.
- Refresh preserves selection, open dialogs, applied inference values, and editor drafts. A temporarily unready/missing session does not erase settings for that same selection.
- Serving status is separate from capability dependencies. `is_model_deployed=False` cannot distinguish loading from missing weights, so **Finish setup** opens the serving session. Live dependency enforcement remains AT-03/AT-06 follow-up work.
- Inference settings retain the prior editability contract: detector YAML is editable; other models' defaults remain read-only. Apply validates a YAML mapping; unapplied drafts do not affect tracking.

## Stop without new workers

Runtime paths: `src/tracking/request_control.py`, `src/tracking/operation.py`, narrow integration edits in `src/tracking/{track,inference}.py` and `src/main.py`.

- Start/continue endpoints register an operation before their background task runs. Stop sets its cancellation event even while the Track constructor is making requests. Repeated Stop is harmless.
- Each operation owns a separate SDK API instance, preserving auth headers and additional fields without mutating the shared app API. Its synchronous POST transport uses one attempt with a **5-second connect timeout** and **30-second read timeout**.
- `AUTO_TRACK_READ_TIMEOUT_SECONDS` is an optional positive integer override, default **30**. This affects API/model reads and the task-request timeout payload. Terminal/error notifications use a **5-second read timeout**.
- Direct tracker URLs and the detector Session's HTTP calls use the same transport limits. The bounded detector Session keeps SDK annotation conversion. Cloud billing requests use the payload contract from the repository's `gpu-cloud` SDK branch and receive the same timeouts.
- Cancellation is checked before subsequent API requests and before consuming/committing returned predictions. Detector prefetch no longer starts a detached thread. Existing geometry/Smart Tool execution threads were not expanded or replaced.
- A model read timeout ends the operation with an actionable message. Timed-out uploads are not retried as smaller batches. Model results arriving after Stop are discarded. Registry cleanup happens on initialization failure, inference failure, cancellation, and normal completion.
- The toolbox's `notify_progress()` stopped response is now honored. Final notification is owned by operation cleanup. The existing terminal `current=total` signal remains until AT-04 defines the coordinated exact-progress contract; it is not evidence of completed work on cancellation.

### Limits and follow-up verification

HTTP read timeout measures socket inactivity, not a hard total wall-clock deadline. DNS resolution, a peer continuously sending bytes, CPU work, lock contention, and requests already admitted before cancellation are not preempted. There is no promise of instantaneous Stop or a strict 30-second operation-wide bound. Final notification can require another short HTTP call.

Local timeout does not stop remote GPU computation. The legacy synchronous `track-api` has no job ID to cancel. An annotation write already accepted remotely is not rolled back. Cloud billing response ambiguity and unconsumed reservations still require live accounting reconciliation checks; requests are not blindly retried.

The new transport is limited to tracking operations. Other app routes (for example standalone cache/interpolation) are outside this change.

Bidirectional tracking remains planned: sequentially submit two independent directional jobs, keep both IDs, poll/cancel them separately, and aggregate results in Auto Track. It requires the direction-field fix and per-serving state-isolation checks documented in [INFERENCE_RESEARCH.md](INFERENCE_RESEARCH.md).

## Checks

- `tests/test_model_setup.py`: draft versus applied configuration, empty-catalog deployment, conditional interpolation, exact deployed-session selection, failure/readiness transitions, YAML validation, and refresh preservation.
- `tests/test_tracking_stop.py`: cancellation before initialization, bounded task/detector transport without retries, late-result rejection, blocked annotation writes, initialization cleanup, progress-triggered Stop, bounded billing, and reuse of a completed operation ID.
- `tests/check_http_timeouts.py`: an actual localhost HTTP server intentionally sends no response; a shortened test timeout verifies the transport behavior without credentials or annotation writes.
- Browser: real local app table/order, empty catalog, interpolation Apply; simulated serving with real SDK widgets for model-name clicks, editing, Apply, and reopening.
- Test runtime: local SDK **6.74.25**; development requirements point to **gpu-cloud**. Live GPU deployment, platform toolbox Stop, actual inference, and cloud billing are not verified by these local checks.
