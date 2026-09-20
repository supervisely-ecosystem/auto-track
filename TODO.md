# Auto Track — Implementation Tracker

Updated: 2026-09-20. Source: [issue #6171](https://github.com/supervisely/issues/issues/6171) and the user's decisions in this task.

This file records the agreed work and investigation findings. An unchecked item is not implemented. Mark an item complete only after its acceptance checks pass; add the commit/PR and verification evidence to the delivery log below.

| ID | Work | Status | Scope |
|---|---|---|---|
| AT-01 | Remove routine off-screen warnings | Approved; pending | Small fix |
| AT-02 | Replace two generated tags with one global tag | Dependency review completed; compatibility gate remains | Output metadata change |
| AT-03 | Reject detection tracking without a suitable tracker | Explicitly deferred by user | Configuration validation + toolbox integration |
| AT-04 | Report exact frame progress | Approved; pending | Progress accounting refactor |
| AT-05 | Make Stop responsive and reliable | Timeout/cancellation implementation verified locally; live serving validation pending | Track lifecycle + request execution |
| AT-06 | Clear model setup UI with working Deploy flow | Table and unified configuration implemented locally; live deployment validation pending | App UI + deployment lifecycle |
| AT-07 | Continue detection through empty video intervals | Backlog from initial assessment | Frame scheduling refactor |
| AT-08 | Backward and bidirectional tracking | Backlog from issue | Tracking core + serving contracts |
| AT-09 | SAM engine warning | Reported fixed upstream | No implementation planned here |

## AT-01 — Routine off-screen warnings

- [ ] Remove toolbox warning notifications for expected disappearance events; keep diagnostic logs with object, frame, and reason.
- [ ] Preserve warnings/errors for actual inference and configuration failures.
- [ ] Verify area-shrinkage and motion-anomaly cases: tracking still stops as before, without repeated toolbox toasts.

Owner: `Timeline.filter_for_disappeared_objects()` in [track.py](src/tracking/track.py). This task changes notification behavior, not the disappearance heuristics.

## AT-02 — One global tag: compatibility first

User decision: combine the auto-detected marker and confidence only if dependent behavior is preserved.

Findings from the app and a full Python-source search of the local SDK checkout:

- `auto-detected-object` is created/reused by this app; no occurrence was found in the checked SDK. Auto Track does not read this saved marker to continue tracking.
- `confidence` / `conf` **are meaningful SDK conventions**. Consumers include `nn/tracking/boxmot.py`, `nn/tracker/botsort_tracker.py`, `nn/model/prediction.py`, inference filtering, and detection metrics/benchmarks. These findings concern prediction labels; they do not prove that these consumers read Auto Track's saved video-object tags.
- Auto Track reads incoming detector-label confidence in `init_timelines_from_detections()`. It writes a global marker plus a confidence tag limited to the detection frame. It does not subsequently use those saved tags to manage timelines.
- Both current tag definitions omit explicit applicability/scope. The inspected SDK supports `TagApplicableTo.OBJECTS_ONLY` and `TagTargetType.GLOBAL`.

- [x] Inspect app and SDK dependencies; record the scope of the evidence.
- [ ] Confirm compatibility against the deployed SDK and relevant video annotation/export consumers before changing output.
- [ ] Prefer retaining the standard numeric name `confidence` for the single global object tag, with the value from the object's initial detection. Keep incoming detector `confidence` / `conf` parsing intact. This is the proposed design, not an implemented schema.
- [ ] Verify that dropping the separate marker preserves the required provenance semantics; a generic confidence tag can also originate from other tools.
- [ ] Handle existing tag definitions and existing frame-based values explicitly. Do not globally change a shared project's tag scope or delete unrelated `confidence` data. Define migration/conflict behavior before implementation.
- [ ] Verify a fresh project and an existing project: one generated global object tag, correct confidence, no extra generated frame-based tag, unchanged tracking behavior.

Owner: `Track.init_timelines_from_detections()` in [track.py](src/tracking/track.py).

## AT-03 — Detection must require a compatible tracking configuration

Reported reproduction: start Track by Detection on an unannotated frame with a detector configured but no bounding-box tracker. A few cars receive boxes and two tags, but the boxes are not propagated. The supplied screenshot records the resulting annotations; the missing-tracker diagnosis is the user's reproduction context.

This is distinct from AT-07: detection can succeed on the first frame while tracking is unavailable.

- [ ] Validate detector readiness **and** tracker readiness for its output geometries before creating objects or figures.
- [ ] For the reported box case, require a configured, ready bounding-box serving; fail with an actionable message if it is missing.
- [ ] Make toolbox availability reflect the same capability check; retain server-side validation for stale UI/direct API calls.
- [ ] Reconcile `trackingByDetection` written by the endpoint with `trackByDetection` read by the executor.
- [ ] Verify missing, stopped, unready, incompatible, and ready tracking sessions. Invalid starts must produce no partial annotations; a ready configuration must detect and propagate objects without a manual seed.

Owners: [main.py](src/main.py), settings/capabilities, `Track.is_detection_enabled()`, `validate_timelines()`, and initialization in [track.py](src/tracking/track.py). Toolbox disabling may require web changes. **Keep deferred until scheduled.**

## AT-04 — Exact progress over the requested frames

Current behavior aggregates mutable tracklet estimates: frame span × current figure count. It does not maintain an authoritative count of completed frames. New detections, edits, disappearing figures, and range extensions change those estimates. `Progress.notify(stop=True)` also reports `current = total`, including on cancellation; completion accounting needs to be separated from termination signaling.

- [ ] Define the progress unit explicitly as completed video frames for the current requested operation. Aggregate requested ranges without counting overlap twice; a frame is complete only when all applicable work for that operation has finished and its results have been committed.
- [ ] Maintain one authoritative progress owner shared by inference, updates, and result commits. Distinguish completed, skipped, failed, and cancelled work.
- [ ] Give extensions and re-tracking an explicit new/revised work scope. Report its real numerator and denominator; do not hide changes with `max(previous_progress, current_progress)`.
- [ ] Make frame ranges and counts in each notification describe the same snapshot; reject stale notifications/results from earlier operations.
- [ ] Preserve the toolbox's terminal-event contract while ensuring Stop/error is not presented as successful 100% completion. Verify whether a coordinated web/API change is necessary.
- [ ] Verify multiple objects on the same frames, overlapping ranges, disappearance, new detections, edits, extension, partial upload failure, cancellation, and completion. Assert exact counts, not just monotonic percentages.

Owners: `Progress`, tracklet counters, `refresh_progress()`, update handling, and result upload in [track.py](src/tracking/track.py). Coordinate with AT-05 and AT-07.

## AT-05 — Responsive Stop and bounded request execution

Pre-change request path (retained as diagnosis evidence):

1. `/track` schedules a FastAPI background task; the track is registered only after its constructor finishes API calls and possible initial detection.
2. `Track.run()` schedules batches of up to eight frames. Geometry-specific inference runs in a `ThreadPoolExecutor`; the caller waits on `Future.result()`.
3. URL tracking uses synchronous `requests.post(..., timeout=60)`. Session tracking uses synchronous `api.task.send_request(..., "track-api", retries=1)`.
4. Detection uses synchronous `Session.inference_video_id()`. In the inspected SDK, its HTTP helper does not supply a timeout. The task-request timeout is a server payload field; the inspected SDK's outer `Api.post()` does not supply an HTTP timeout either. These are candidate unbounded waits, not a reproduced diagnosis of the reported hang.
5. `/stop_tracking` sets a Boolean flag. It cannot find a track still initializing, does not interrupt a pending model call, and the loop may upload returned predictions before checking the flag again.
6. SDK `video.notify_progress()` returns a stopped indicator that the app currently ignores.

- [x] Trace current request/stop paths and inspect SDK timeout/cancellation interfaces.
- [ ] Reproduce and measure Stop during initialization, detector inference, tracker inference, upload, and idle waiting; define and verify an explicit local cancellation-latency target.
- [x] Register cancellable operations before blocking initialization. Treat repeated Stop as idempotent and honor the SDK's returned stopped indicator.
- [ ] On Stop, prevent new work and discard late results for that operation. Gate annotation mutations and progress updates; define reconciliation for a write already accepted by the server.
- [ ] Keep annotation writes, track state, and billing reconciliation under one parent owner. Do not allow a detached worker to upload predictions after cancellation.
- [x] Bound model/API HTTP waiting without introducing threads or subprocesses. Use a per-operation API client, 5-second connect timeout, configurable read timeout (30 seconds by default), and one transport attempt. Keep terminal-notification reads at 5 seconds. Suppress late inference results and gate later API calls after Stop. See [implementation notes](IMPLEMENTATION_NOTES.md) for the precise limits.
- [ ] Do not move the entire mutable `Track` into a killable subprocess: abrupt termination during writes/billing would leave partial state. Do not rely on `Future.cancel()` to stop an already running Python thread.
- [ ] Check per-serving remote cancellation. The inspected SDK exposes `inference_video_id_async()` and `stop_async_inference()` for supported inference sessions, but that does not establish cancellation support for every `track-api` serving.
- [ ] Verify no stale local commits/progress after cancellation, no worker leaks, successful subsequent tracking, and bounded local completion when the remote model never replies.

Decision: do not introduce subprocesses or new worker threads for Stop. Existing geometry inference threads remain; detached detector prefetch was removed. Timeout/cancellation checks cover initialization, SDK task requests, detector session calls, direct model URLs, annotation API requests, and cloud billing HTTP calls. A live serving and cloud billing run still need validation.

A subprocess can stop **local waiting**; killing it does not by itself cancel remote GPU computation. Keep these two guarantees separate. Do not stop a shared serving session to cancel one track.

Owners: [main.py](src/main.py), [track.py](src/tracking/track.py), [inference.py](src/tracking/inference.py), and serving/SDK contracts where necessary.

## AT-06 — Clear model setup UI with working deployment

User correction: simplify presentation so Auto Track feels like a usable application and makes model setup for each geometry obvious. **Keep and repair Deploy model.** This is not a request to restrict the product to already deployed sessions or remove existing capabilities.

Observed in the screenshot: large repeated cards, a connection-mode selector above each model selector, mostly empty YAML editors occupying most of the width, repeated technical copy, and no clear setup overview.

Current implemented layout:

- Header: “Set up Auto Track” with a short instruction to choose or deploy a model for each geometry the user wants to track. Do not imply that every optional geometry must be configured.
- A compact table shows geometry/purpose, selected model, status, and Configure/Change. A ready model name opens inference settings. Configure combines session selection and **Deploy model** in one dialog, with **Open session / Finish setup** where needed.
- Purpose descriptions explain optional detection and Smart Tool dependencies; live dependency indicators/enforcement remain pending.
- Open inference settings in a dedicated modal by clicking the ready model name. Sessions, deployment, supported interpolation, and conditional URL configuration live inside Configure; no connection-mode selector is shown on the main screen. Disappearance tuning remains collapsed.
- Show serving readiness and the next required action. Capability-level validation for detection tracking remains AT-03; serving readiness alone does not establish it.

Pre-change findings in `src/ui/classes.py` (addressed by the local implementation):

- `update_nn()` explicitly hides an open deploy dialog. `_auto_refresh()` currently sleeps 60 seconds; the issue reports approximately five seconds, so the deployed timing still needs verification. The close-on-refresh behavior exists independently of that timing.
- `deploy()` starts a session and waits for application `STARTED`, which does not mean model weights are deployed. It invokes a generic refresh callback without passing the newly created session ID for explicit selection.
- `update_nn()` checks `is_model_deployed()` but only shows a generic waiting message when false; it provides no guided handoff to finish serving configuration.
- Refresh replaces inference-editor contents with defaults, and the visible Save handler is currently `pass`.

- [x] Implement the compact setup overview and geometry controls; remove duplicate technical explanations and always-visible empty editors.
- [x] Keep Deploy model and hand the created session ID to the initiating control without waiting for model initialization. Verify the dispatch contract with a fake serving API.
- [x] Distinguish starting, ready, unready, unavailable, and failed deployment. Since an unready flag cannot identify loading versus missing weights, direct the user to inspect/finish setup in the serving session.
- [x] Select the exact newly created session in the initiating geometry control.
- [x] Add **Finish setup / Open session** links and refresh readiness. Verify unready → ready transition with a fake serving API.
- [x] Preserve open deployment dialogs, model selections, applied settings, and editor drafts across refresh.
- [x] Open inference settings in a modal; validate YAML mappings on Apply and keep unapplied drafts out of runtime settings. Preserve existing read-only behavior for non-detector settings.
- [x] Offer interpolation only for supported geometries inside the configuration dialog. Preserve conditional URL support without a redundant mode dropdown on the table.
- [ ] Show actionable dependency messages for Smart Tool and Track by Detection; coordinate enforcement with AT-03.
- [x] Browser-check first launch, compact layout, interpolation selection, and Deploy dialog with no GPU agent against the real local app. Check the inference modal, Apply, reopening, and Refresh using real SDK widgets with a simulated ready serving.
- [ ] Complete live serving verification: automatic deployment, manual weights, deployment failure, and actual inference with the chosen settings. No compatible running serving/GPU agent was available in the fallback team during this pass.

Owners: [ui/model_row.py](src/ui/model_row.py), [ui/model_settings.py](src/ui/model_settings.py), [ui/classes.py](src/ui/classes.py), [ui/common.py](src/ui/common.py), [ui/ui.py](src/ui/ui.py). This app's Deploy dialog is in scope. Making Auto Track discoverable/openable directly from the video toolbox still needs web integration; improving this app alone does not remove that navigation problem.

### Table UI — implemented after user feedback

The user prefers a compact table with existing geometry/purpose icons, priority ordering, a separate status widget, and clickable model names for inference settings. The table replaces the first compact-card iteration above.

- [x] Replace repeated cards with columns: **Purpose / Model / Status / Action**. Reuse platform geometry icons; keep a visible purpose label and accessible labels for icon buttons.
- [x] Use the user-approved order: **Bounding Box → Point-based → Smart Tool → Detector → Mask → Oriented Box**. The detector remains optional for manual-seed tracking.
- [x] Display the selected model as a settings link when ready, plain text when unavailable, or “—” when not configured; remove permanently empty session selectors and standalone deployment buttons from the table. Use one **Configure / Change** action per row.
- [x] Open one model-configuration dialog for the selected purpose. Offer compatible existing sessions and **Deploy new model** inside the same dialog; use a single selection form with no Existing/Deploy tabs. **Deploy new** expands deployment inline without hiding the session picker. Empty catalogs show an explanation and the deploy action. Do not stack a second deploy modal.
- [x] Show **Interpolation** only for supported geometries. Keep URL configuration conditional on the existing environment rule and inside the configuration dialog, rather than a selector on every row.
- [x] On selection, commit explicitly with **Use model**; closing an unfinished selection keeps the previously applied configuration. On deployment, select the exact created session and show its startup/setup state. Do not silently undo an already launched remote session if the dialog is dismissed.
- [x] Use distinct status icons with short text: **Not configured**, **Starting**, **Needs setup**, **Unavailable**, **Ready**. Do not infer “model loading” versus “needs weights” from `is_model_deployed=False` alone; offer **Open session** to resolve the ambiguity. Readiness reflects the serving check, not a promise of tracking quality.
- [ ] Show unmet capability dependencies as an actionable hint (for example, detector ready but no box tracker), separately from serving readiness. Avoid marking optional unconfigured rows as application-wide errors.
- [x] Open inference settings by clicking the ready model name; remove the separate Settings column. Use plain model text when settings are unavailable; keep Apply/draft/refresh behavior intact.
- [ ] Verify empty catalog, existing ready session, interpolation, deployment, required manual setup, unavailable session, dependency hints, and cancelled configuration edits in the browser.

Implemented with the repo-local `ModelTable` widget and SDK controls. Styling now comes from the platform's `el-table` / `ultra-table` and SDK `Text` status styling, without a separate table CSS file. Browser-checked table order, empty-catalog deployment, interpolation Apply, and model-name inference settings/Apply/reopening. Ready serving state is simulated for the inference-modal check; no GPU deployment was performed.

## AT-07 — Detection through empty intervals

- [ ] Give detection its own cursor over the requested video range, independent of existing object timelines.
- [ ] Continue scanning when the first frame contains no detections or all tracked objects disappear; create timelines when objects enter later.
- [ ] Integrate exact progress and cancellation with AT-04/AT-05.
- [ ] Verify empty prefix, fully empty range, disappearance followed by new arrivals, and no duplicate objects.

Owner: `Track.get_batch()`, initialization, and the main loop in [track.py](src/tracking/track.py).

## AT-08 — Backward and bidirectional tracking

Proposed execution model: one parent operation owns two directional passes from the same manual seed. It sends bounded inference requests and alone commits annotations, progress, and billing. Workers return predictions with explicit frame indexes and an operation revision; they do not own shared mutable timelines or upload annotations.

Example: seed frame 100; backward predictions cover 99 → 80 and forward predictions cover 101 → 120. The seed is not uploaded twice; for this fixed range there are 40 target frames. Sequential batches within each direction preserve dependencies on previous predictions. The two directions may run concurrently when the serving supports independent request state and has capacity, or be interleaved otherwise.

User decision: use HTTP timeouts and cancellation checks for Stop (AT-05), without a new worker/subprocess execution layer. For bidirectional tracking, submit two directional requests and manage their results in the parent operation. This bidirectional orchestration remains planned, not implemented. They do not add backward support to a model, isolate state inside the remote serving, or guarantee parallel GPU execution. Async I/O can use the same orchestration model. A backward pass must feed frames in reverse temporal order; reversing forward predictions is not equivalent.

If a serving accepts jobs asynchronously and immediately returns job IDs, Auto Track can submit both jobs sequentially without waiting for the first result. This avoids local concurrent submissions, with speed depending on submission overhead and serving scheduling. A serving without a direction argument could still support reverse tracking if its frame/file interface accepts frames supplied in reverse temporal order; otherwise that serving needs an interface/implementation change.

The [YOLOv8 / SDK source audit](INFERENCE_RESEARCH.md) confirmed that the server's async video endpoint returns a UUID after scheduling, but `Session.inference_video_id_async()` waits for startup and stops its previous job on reuse. The checked SDK defaults to a multi-worker executor; this does not establish model-level concurrency or unchanged throughput. Bbox/mask/point tracking base interfaces already support backward. The detector client sends `framesDirection` while the active server reads `direction`, so the high-level backward argument is ignored in the checked version. Separate UUIDs also do not isolate a tracker's shared model state.

- [x] Audit YOLOv8 and SDK async submission and backward frame traversal; record exact source versions and limitations.
- [ ] Correct the detector direction-field contract and verify against the deployed SDK/serving version.
- [ ] Separate async job submission, polling, and cancellation by UUID; avoid reusing the single-job `Session` wrapper for both directions.

- [ ] Audit direction support for each serving and propagate direction through the inference contract.
- [ ] Verify each serving's concurrent-request/state-isolation guarantees and resource limits before submitting both passes concurrently. Use separate request identities and a parent operation revision; ignore stale results after edits or cancellation.
- [ ] Make batch traversal, prediction frame indices, keyframe boundaries, `no-objects` intervals, edits/deletions, and uploads direction-aware.
- [ ] Coordinate two passes for bidirectional tracking, including shared seed ownership, cancellation, and exact progress.
- [ ] Verify partial completion and failure of one direction without losing successful work or reporting overall success; Stop must cancel both passes locally.
- [ ] Verify video boundaries, manual keyframes, excluded intervals, multiple geometries, multiview isolation, and unsupported serving capabilities.

The current core is forward-oriented; changing the toolbox direction control alone is insufficient. Existing interpolation direction support does not establish model-tracking support.

## AT-09 — Already reported fixed

The issue says the SAM engine warning was fixed by Stas. Treat this as an upstream report, not a locally verified fix. Check it when testing the relevant deployed version; do not duplicate the implementation task unless it reproduces.

## Evidence and delivery log

- 2026-09-20: Created this tracker and completed static app/SDK dependency and request-path inspection. No runtime changes or live reproduction in this planning pass.
- 2026-09-20: Re-read issue #6171 after the user's UI clarification. Replaced the incorrect session-only/removal-of-Deploy plan with a simplified UI that retains and repairs deployment. Added a proposed two-pass bidirectional execution model and its serving constraints. Runtime remains unchanged.
- App inspected at `8377f49c707d07bd24a95a3b2c794f94860896f1`.
- SDK inspected in the sibling `../supervisely` checkout, branch `master`, commit `ed04d3af8bb34fdc474d4178945bb30be5741250`.
- 2026-09-20: Implemented AT-06 locally. Verification: `.venv/bin/python -m unittest discover -s tests -v` — 9 passing tests; `git diff --check`; local Uvicorn app and browser checks described above. No production deployment or GPU model launch performed.
- Runtime caveat: created an isolated repo-local `.venv` with SDK 6.74.25 built from the inspected local checkout, plus the existing app's runtime dependencies. Development requirements reference SDK branch `gpu-cloud`, so deployed-runtime parity still needs checking. `local.env` selected an inaccessible team; browser validation used verified fallback team 9 / workspace 8 without modifying their annotations.
- 2026-09-20: Implemented the approved table order and unified configuration dialog as a local widget. Implemented timeout-based cancellation without new runtime worker threads or subprocesses. Local verification: 22 unit tests passed; a real local silent HTTP endpoint triggered a 0.1-second test read timeout after 0.106 seconds; compileall and diff checks passed. Browser checks covered table rendering, interpolation Apply, and reopening saved inference settings. Full GPU serving, cloud accounting, and toolbox Stop end-to-end remain unverified.
- Delivery entries: record task ID, commit/PR, verification command or browser scenario, result, and remaining limitations here.

- 2026-09-20: Matched table corners to native cards (5 px), moved inference settings to model-name clicks, and replaced configuration tabs with one selector plus inline Deploy new. Refresh/cancel preserve the session draft. Fixed queued-serving status rendering found during checks.
