<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/auto-track/assets/119248312/c372ff77-4025-4e55-80b8-e821e31868e3"/>  

# Auto Track

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/auto-track)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/auto-track)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/auto-track.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/auto-track.png)](https://supervise.ly)

</div>


## Overview

This app offers a convinient way to apply tracking to the objects on video and speed up annotation proccess. When annotating video, this app will automatically track objects on next frames based on annotated object. Also, with this app you can track objects of different geometries at the same time and use "Smart Tool" to annotate masks with the help of Neural Network.

## How To Use

### 1: Run the app from ecosystem

<img src="https://github.com/supervisely-ecosystem/auto-track/assets/61844772/c38b4d5f-602d-443e-b345-9b2e6b98bc3c"/>

### 2: Select NN model app for geometries you want to track

To track masks, annotated with `Smart Tool` you will need to select a model for the following geometries: `Bounding Box`, `Point` and `Smart Tool`.

<img src="https://github.com/supervisely-ecosystem/auto-track/assets/61844772/ee1158d7-d9f4-4761-b514-c3f25fc6c8e5"/>


### 2.1: You can also deploy model from the app UI. Click on the `Deploy new model` button, in the appeared modal window select the model, agent and extra parameters if needed. After that click on the `Deploy model` button. The task will be started and selected for the tracking.

<img src="https://github.com/supervisely-ecosystem/auto-track/assets/61844772/31312e90-ae88-45fd-b86c-48c6e0b78e15"/>


### 3: Select Auto-Track appliction in the Video Annotation Tool

<img src="https://github.com/supervisely-ecosystem/auto-track/assets/61844772/09849a43-5d4b-42d4-8b94-f527c41cebea"/>

## Key features

1. After you create or edit an object, the app will automatically track the object on next frames. You can also click on the `track` button to start tracking on all the objects in the frame.
2. When you iterate through the frames, the app will automatically continue track when you reach last frames of the tracking interval.
3. If an object is occluded for single frame, you can delete it on that frame. To do it you need to select it on the frame, click the right mouse button and select `Delete object`. The app will stop tracking the object on this frame.
4. If an object has left the scene, you can delete it permanently. To do it you need to select it on the frame, click the right mouse button and select `Delete all from current`. The app will stop tracking the object until the end of the video.

<img src="https://github.com/supervisely-ecosystem/auto-track/assets/61844772/bb29ee0f-27c3-4fe3-84d8-0cb5c116bed3"/>

5. You can use Auto-Track app as a `Smart Tool` if you have selected a model for the `Smart Tool` geometry. Tracking objects, annotated with `Smart Tool` requires a model for the following geometries: `Bounding Box`, `Point` and `Smart Tool`.
