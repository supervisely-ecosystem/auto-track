# Auto Track — Knowledge Map

Auto Track is Supervisely’s video annotation orchestration app: it routes annotated objects to tracking models, manages tracking state, and writes predictions back to the platform.

| Area | Key knowledge |
|---|---|
| Purpose | Reduce manual labeling by propagating object annotations across video frames. |
| Workflow | Configure models → annotate an object → start tracking → review/correct predictions → extend or stop tracking. |
| Core concepts | **Object:** persistent identity. **Figure:** geometry on one frame. **Key figure:** manual annotation. **Track:** tracking session for one video. **Timeline / tracklet:** per-object tracking intervals. |
| Model routing | Boxes: MCITrack, MixFormer, SAM3. Masks: XMem, SAM2, SAM3. Points, polylines, polygons, skeletons: CoTracker. Smart Tool: ClickSeg, SAM2, SAM3. Detection: YOLOv8. |
| Editing behavior | Responds to annotation changes, track extension, object removal, and `no-objects` tags; updates affected tracking intervals. |
| Additional capabilities | Interpolation, detection of new objects, disappearance filtering, interactive segmentation, model cache warm-up. |
| Runtime | Python, Supervisely SDK UI, FastAPI/Uvicorn on port 8000. Inference runs through external model URLs or Supervisely app sessions. |

The toolbox controls automatic retracking, frame ranges, direction, and continuation near the end of a tracked segment. See the [Videos 3.0 documentation](https://docs.supervisely.com/labeling/labeling-toolbox/videos-3.0#auto-tracking).

In **multiview**, a dataset groups synchronized camera videos, with shared object identities and video-specific figures and tags. Users seed the same object in additional views; Auto Track processes video-specific tracking sessions. See the [multiview documentation](https://docs.supervisely.com/labeling/labeling-toolbox/multi-view-videos#auto-tracking).

## Code Navigation

- [src/main.py](src/main.py) — toolbox API endpoints and background dispatch.
- [src/ui/ui.py](src/ui/ui.py) — model selection and tracking settings.
- [src/globals.py](src/globals.py) — model registry, geometry mappings, active tracks.
- [src/tracking/track.py](src/tracking/track.py) — tracking lifecycle, timelines, updates, prediction uploads.
- [src/tracking/inference.py](src/tracking/inference.py) — model requests and Smart Tool orchestration.
- [src/tracking/interpolation.py](src/tracking/interpolation.py) — interpolation between annotations.

Based on repository inspection and the linked documentation on 2026-09-20; runtime behavior has not been tested.
