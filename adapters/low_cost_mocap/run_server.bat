@echo off
rem ==================================================================
rem  run_server.bat - Low-Cost-Mocap realtime-video adapter
rem
rem  Replays a multi-camera video source as a live stream, served by
rem  Low-Cost-Mocap's own pipeline (helpers.py + index.py) with only
rem  three substitutions:
rem    * video source  : VideoReplay instead of pseyepy.Camera
rem                      (clips / RTSP / MJPEG / USB / PNG dirs, see
rem                       video_replay.py -- MOCAP_SOURCES)
rem    * multi-target  : locate_bodies() when num_objects > 1 -- the
rem                      upstream triangle matcher run once per body
rem                      signature (see multi_body.py)
rem    * drone link    : null serial stub (no real drone / no PS3-Eye)
rem
rem  MJPEG stream ->  http://localhost:3001/api/camera-stream
rem ==================================================================
setlocal
set ROOT=%~dp0..\..
set VENV=%ROOT%\.venv\Scripts\python.exe
set API=%ROOT%\adapters\low_cost_mocap\low-cost-mocap\computer_code\api

rem ---- adapter mode: visible-light marker detection ---------------------
if "%MOCAP_VISIBLE_MODE%"=="" set MOCAP_VISIBLE_MODE=1
if "%MOCAP_DOT_THRESHOLD%"=="" set MOCAP_DOT_THRESHOLD=140
if "%MOCAP_DOT_MIN_AREA%"==""  set MOCAP_DOT_MIN_AREA=3
if "%MOCAP_DOT_MAX_AREA%"==""  set MOCAP_DOT_MAX_AREA=200

rem ---- video source ------------------------------------------------------
rem  Default: the rendered dataset clips.  For real hardware, e.g.:
rem    set MOCAP_SOURCES=rtsp://10.0.0.11:8554/cam1,rtsp://10.0.0.12:8554/cam2,...
rem    set MOCAP_SYNC_MODE=native
if "%MOCAP_VIDEO_DIR%"==""  set MOCAP_VIDEO_DIR=%ROOT%\out\video
if "%MOCAP_VIDEO_TAG%"==""  set MOCAP_VIDEO_TAG=t650
if "%MOCAP_CALIBRATION%"=="" set MOCAP_CALIBRATION=%ROOT%\out\rig3\cameras_calib.json

rem ---- rig geometry + autoscale for the single-body matcher --------------
if "%MOCAP_AUTOSCALE%"==""   set MOCAP_AUTOSCALE=0
if "%MOCAP_RIG_DIST1%"==""   set MOCAP_RIG_DIST1=0.0696
if "%MOCAP_RIG_DIST2%"==""   set MOCAP_RIG_DIST2=0.0470
if "%MOCAP_RIG_TOL%"==""     set MOCAP_RIG_TOL=0.020

rem ---- multi-target: N rigid bodies with distinct marker signatures ------
rem  index.py sets num_objects; MOCAP_BODY_SPECS points at the rig file that
rem  carries one (dist1, dist2) triangle per drone.
if "%MOCAP_NUM_OBJECTS%"=="" set MOCAP_NUM_OBJECTS=3
if "%MOCAP_BODY_SPECS%"==""  set MOCAP_BODY_SPECS=%ROOT%\out\rig3\rig_blender.json
if "%MOCAP_BODY_TOL%"==""    set MOCAP_BODY_TOL=0.025
rem  Temporal association (timestamped constant-velocity gate) and the rigid
rem  identity check on non-reference markers.  gate=0 disables association.
if "%MOCAP_BODY_GATE_M%"==""      set MOCAP_BODY_GATE_M=0.5
if "%MOCAP_BODY_EXTRA_TOL%"==""   set MOCAP_BODY_EXTRA_TOL=0.05
if "%MOCAP_REACQUIRE_GAP_S%"==""  set MOCAP_REACQUIRE_GAP_S=0.5
if "%MOCAP_SOURCE_FPS%"==""       set MOCAP_SOURCE_FPS=30
if "%MOCAP_BODY_MAX_ERR%"==""      set MOCAP_BODY_MAX_ERR=6.0
if "%MOCAP_MAX_SPEED_MPS%"==""     set MOCAP_MAX_SPEED_MPS=10.0
if "%MOCAP_POSITION_REFERENCE%"=="" set MOCAP_POSITION_REFERENCE=body_origin

cd /d "%API%"
"%VENV%" index.py
endlocal
