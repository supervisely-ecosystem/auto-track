# Findings to retain

Updated 2026-09-20. Evidence and immutable code links: [INFERENCE_RESEARCH.md](INFERENCE_RESEARCH.md). Delivery status: [TODO.md](TODO.md).

| ID | Finding | Consequence / status |
|---|---|---|
| SDK-01 | Video inference Session writes `framesDirection`; the active SDK server reads `direction`. | `backward` becomes forward in the inspected version. Reproduced from the state builder; SDK fix and live-version verification remain pending. |
| SDK-02 | `Session.inference_video_id_async()` waits for startup and holds one mutable request UUID. Reusing it stops its previous job. | Use separate job IDs and submission/polling/cancellation operations for bidirectional work; do not submit both through one Session instance. |
| SDK-03 | `task.send_request(timeout=...)` sets a server payload field; the checked synchronous `Api.post()` and Session model transport do not set a client HTTP timeout. | Local tracking now supplies explicit connect/read timeouts. Upstream SDK behavior is unchanged. |
| SDK-04 | Tracker base classes support backward traversal, but models can share mutable state initialized per request. | Two UUIDs do not guarantee isolated or safe concurrent tracking. Audit chosen serving implementations before bidirectional execution. |
| APP-01 | Auto Track plans/indexes frames forward and previously omitted direction from model requests. | Backward needs an app scheduling/result-mapping refactor; it is not a UI-only switch. |
| APP-02 | Detection endpoint writes `trackingByDetection`, while the executor reads `trackByDetection`. | Preserve in AT-03 configuration/behavior repair; not changed by the UI or Stop implementation. |
| SDK-05 | `gpu-cloud` billing methods issue their own `requests.post()` calls without timeouts. | Local tracking now sends bounded billing requests with the same payloads; live accounting reconciliation remains unverified. |

SDK-01–04 refer to SDK `ed04d3af8bb34fdc474d4178945bb30be5741250`. SDK-05 was inspected at local `gpu-cloud` commit `64149a3df3075c0e8c9b2de338f47f375e4b93a4`, file `supervisely/api/cloud.py`. These findings do not establish which SDK version a deployed serving is running.
