# Low-Cost-Mocap 实时视频适配器

把本仓库渲染的 6 机位 OptiTrack 风格视频（`out/video/0?_cam0?_t650_20s.mp4`）
作为 **实时多相机输入** 喂给 [jyjblrd/Low-Cost-Mocap](https://github.com/jyjblrd/Low-Cost-Mocap)
的动捕管线，用它原本的图像处理 + 点检测 + 跨相机极线匹配 + 三角化 + 物体融合，
模拟"实时动捕"运行。

仓库本体已 **vendor** 在 `adapters/low_cost_mocap/low-cost-mocap/`（MIT 许可），
改动集中在 `computer_code/api/helpers.py` 与 `index.py`，原始文件备份为
`helpers.orig.py` / `index.orig.py`。

## 为什么需要适配

Low-Cost-Mocap 后端 `helpers.py` 的唯一视频源是 `pseyepy.Camera`
（读真实 PS3-Eye 摄像头）。本场景**没有摄像头**，只产出了 6 段同步视频。
适配器把该视频源换成"视频回放"——逐帧同步读出 6 路画面、按源帧率
按视频帧号提供 30 fps 捕获时间戳。极线搜索、DLT 与 Kalman 采用上游实现，
刚体几何、时间关联和跟踪参考点由本项目适配；实际处理帧率取决于运行设备。

## 目录

```
adapters/low_cost_mocap/
  video_replay.py          视频回放源（VideoReplay，pseyepy 兼容 drop-in）
  run_server.bat           一键启动 MJPEG 流服务器
  low-cost-mocap/          vendor 的 Low-Cost-Mocap 仓库（含我们的补丁）
    computer_code/api/
      helpers.py           已打补丁：视频源、可见光检测、sfm 回退、物体几何
      index.py             已打补丁：串口空壳、numpy 兼容、关闭 debug reload
      helpers.orig.py      helpers.py 原始备份
      index.orig.py        index.py 原始备份
      camera-params.json   由本仓库标定生成（6 台 OV9281 内参）
```

## 运行

当前三机完整误差视频（机体原点参考点）：
[`out/video/fused_tracking_error_body_origin.mp4`](../../out/video/fused_tracking_error_body_origin.mp4)。
其指标来自**实际干净 MP4 的像素检测**，不是理想投影 CSV。
主参考点为机体原点 `(0,0,0)`（下中心板中心的固定几何参考点，非实测质量质心）。
原始位置取刚体拟合的 `pose_t`，时间关联及 Kalman 都在这个点上工作。
输出 `pos/vel/acc` 对应同一点；`base_midpoint_measured` 是底边中点的未滤波诊断值。
定位/测速/姿态汇总见主 README 与 `out/rig3/accuracy3.json`。

独立、可复用的分析入口（不依赖前端、不读取真值）：

```bat
.venv\Scripts\python.exe adapters\low_cost_mocap\run_tracking.py --frames 600
```

更换真实源只更换输入和标定，算法不变，例如：

```bat
.venv\Scripts\python.exe adapters\low_cost_mocap\run_tracking.py --sources "rtsp://CAM01_HOST/feed,rtsp://CAM02_HOST/feed" --calibration real_cameras.json --body-specs real_marker_rigs.json --frames 0
```

示例地址必须替换；相机数量/顺序应与标定 JSON 的 `cameras[]` 一致。
标定字段为 `name,K,dist,R,t`，`R/t` 指定输出世界坐标；标记配置为 `bodies[]` 下的
`body_id,name,markers[{id,local}],apex_marker_id,base_marker_ids`。
标记 `local` 坐标必须相对于希望跟踪的固定机体原点标定。
当前三机为 `out/rig3/` 中对应 JSON；相机视频本身不随参考点选择改变。
通用 RTSP/USB 的时间是 **host-arrival**，软件检查时差不能证明硬件曝光同步；
精确动捕仍需共同触发/PTP 及真实相机时间戳。失败画面不会作为新观测重复使用。

```bat
adapters\low_cost_mocap\run_server.bat
```

启动后在浏览器打开 Low-Cost-Mocap 前端（或直接用浏览器看流）：

- MJPEG 流：  http://localhost:3001/api/camera-stream
- 配置接口：  http://localhost:3001/api/tracking-config
  （当前六机位标定、3 个刚体的 id/名称；前端启动时自动拉取）
- 前端：     `cd computer_code && yarn install && yarn run dev`  （默认 5173，
  或 `npm run build && npm run preview` 使用已构建版本）

前端显示的是 2×3 相机马赛克（每格带机位名、当前亮斑数与时间戳），点 "Start"
开始接收流；点 "Live Triangulation Start" 进入实时三角化，再点 "Locate Objects
Start" 后，"Rig identity & pose" 面板实时显示每架机的锁定状态、
使用标记数（如 4/4）、姿态模式（4-marker / 6-DOF，或 triangle + prior）与
拟合残差。时间连续关联（0.5 m 常速度门 + 姿态分支先验）在该面板表头以
`gate 0.50 m` 标示，同时显示 `Body origin (geometric center reference)` 与
`position + velocity + acceleration (9 states)`，每机列出滤波后的原点 XYZ。

## 适配点（对仓库的最小改动）

1. **视频源替换**（`helpers.py` 顶部）：`from pseyepy import Camera` →
   改为 `from video_replay import VideoReplay as Camera`。`VideoReplay` 提供
   `pseyepy` 兼容接口（`.read()` 返回 `(frames, timestamps)`、`num_cameras`、
   可变 `gain`/`exposure`、`RES_SMALL`）。

2. **可见光检测模式**（`helpers.py`）：默认 IR 高通边缘检测不适合灰底画面，
   新增 `visible_mode`（环境变量 `MOCAP_VISIBLE_MODE=1` 开启）：
   - `_camera_read` 跳过 IR 高通 kernel；
   - `_find_dot` 改用**亮度阈值 + connectedComponents** 找亮斑（标记峰值
     ~216 vs 灰地 ~62），按面积过滤地面噪声。

3. **sfm 模块回退**（`helpers.py`）：pip 版 opencv 无 `cv2.sfm`，
   提供 `fundamental_from_projections(P1,P2)` 闭式实现（`F=skew(e')P'pinv(P)`）
   供实时三角化的极线匹配使用；`cv.sfm` 存在时仍用原生。

4. **物体几何参数化**（`helpers.py`）：`locate_objects` 的标记间距原来硬编码
   （9.5/15 cm，真无人机），改为可配置，并加自动尺度校正
   （`MOCAP_AUTOSCALE`，用已知 47mm 底边反推真实米制）。

5. **串口空壳**（`index.py`）：无真串口时 `ser` 用空实现（`.write` 返回长度），
   兼容有真机时仍能发指令。

6. **numpy 兼容**（`index.py`）：`.tostring()` → `.tobytes()`（numpy≥2 移除前者）。

7. **关 debug reload**（`index.py`）：`use_reloader=False`，避免 Flask 重启子进程
   丢失 MOCAP_* 环境变量。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MOCAP_VISIBLE_MODE` | 1 | 1=可见光亮度检测；0=仓库原 IR 高通链 |
| `MOCAP_DOT_THRESHOLD` | 140 | 亮度阈值（标记过曝 / 灰地~60） |
| `MOCAP_DOT_MIN_AREA` | 3 | 标记最小连通域面积（滤地面噪声） |
| `MOCAP_DOT_MAX_AREA` | 200 | 标记最大连通域面积 |
| `MOCAP_AUTOSCALE` | 0 | 已给定米制标定时禁止再次估计尺度 |
| `MOCAP_VIDEO_DIR` | `out/video` | 视频目录 |
| `MOCAP_VIDEO_TAG` | `t650`（启动脚本） | 只读取该数据集的六段干净片，不混入单机/HUD |
| `MOCAP_CALIBRATION` | `out/rig3/cameras_calib.json`（启动脚本） | 服务器与前端共用的标定来源 |
| `MOCAP_NUM_OBJECTS` | 3（启动脚本） | 刚体数量 |
| `MOCAP_BODY_SPECS` | `out/rig3/rig_blender.json` | 每个刚体的几何模板 |
| `MOCAP_BODY_TOL` | 0.025 | 边长匹配容差（米） |
| `MOCAP_BODY_GATE_M` | 0.5（启动脚本） | 时间关联门限（米，≤0 关闭） |
| `MOCAP_BODY_EXTRA_TOL` | 0.05（启动脚本） | 第 4 标记刚体预测容差（米） |
| `MOCAP_REACQUIRE_GAP_S` | 0.5（启动脚本） | 预测/姿态先验失效时长（秒） |
| `MOCAP_SOURCE_FPS` | 30（启动脚本） | 源帧率（回放节拍与 Kalman 采样率） |
| `MOCAP_BODY_MAX_ERR` | 6.0 | 参考三角形重投影 MSE 上限（px²） |
| `MOCAP_MAX_SPEED_MPS` | 10.0 | 所选参考点的速度测量幅值上限（m/s） |
| `MOCAP_POSITION_REFERENCE` | `body_origin` | 机体原点；`base_midpoint` 可复现旧参考点 |

独立入口对应 `--position-reference body_origin --max-err 6 --max-speed 10`。
Python `TrackingPipeline` 也显式接收这三个参数；`locate_bodies` 的低层默认值保留
`base_midpoint` 以供上游回归测试，应用和独立入口均显式选择原点。

## 验证结果（实测）

- 6 段视频逐帧同步读入，2×3 马赛克 MJPEG 流正常；多窗口/刷新取流经
  `_capture_lock` 串行共享，无 FFmpeg 并发崩溃。
- 原点模式的位姿须有第 4 标记或新鲜时间先验消歧；无确认的三点解记为无解。
  空帧更新失锁状态，回放倒退或循环会重置跟踪纪元。
- 离线全量口径（600 帧）：原点位置均值 A/B/C 3.51/3.37/4.30 mm，
  最大 13.11/8.71/17.01 mm；测速 MAE 0.036/0.043/0.041 m/s，
  覆盖 98.3%/100%/96.5%，详见主 README 与 `out/rig3/accuracy3.json`。
- `tools/verify_body_origin_app.py` 从运行中的 MJPEG/Socket.IO 服务核验原点和 p/v/a；
  该可选验证脚本需 `requests` 与 `python-socketio` 客户端依赖。
- `fundamental_from_projections` 与 OpenCV 一致（真匹配点极线距离 0）。

## 限制

- **60 fps 部署状态**：本机 i7-1255U 中心纯点输入已优化到平均 14.86 ms/组
  （3×600 帧），P95 19.97 ms、P99 23.77 ms。平均计算吞吐提升，但尚不能承诺稳定 60 Hz；
  Web 集中式链路还包含解码、检测和 MJPEG。详见 [中心优化](../../docs/fusion_optimization.md)
  与 [部署评估](../../docs/edge_architecture.md)。
- **优化后精度**：完整视频重新定位的误差直方图、CDF 和 XYZ 分量见
  [误差分布报告](../../docs/position_accuracy_optimized.md)，输出为 `tracking_optimized.*` 与
  `accuracy_optimized.*`。滤波位置逐帧与原基线相同；精度报告中的 30 fps 是采样时间轴。
- **独立采集调度**：当前 Web 应用由 `/api/camera-stream` 请求驱动采集/融合；生产部署
  需要独立生产者和有界队列，监控只消费结果。`MOCAP_SOURCE_FPS=60` 不会配置相机传感器
  为 60 fps，已有 30 fps 文件也不会变成原生 60 fps 数据。
- **坐标系**：DLT 输出由输入 `R/t` 定义。本数据集给出的标定为 ENU、米制；
  `tracking_pipeline.py` 不再叠加坐标轴翻转/自动缩放。前端在线标定仍需按原仓库
  做 floor/origin 校准，不能把其坐标变换混用于已知 ENU 标定。
- **模型二义性**：等腰参考三角形交换底边等价于 180° 旋转，三点无法区分。
  第 4 标记提供几何确认；被完全遮挡时由最近接受姿态的时间先验选支
  （0.5 s 内有效），冷启动/长间断后的三点解仍可能二义，
  由 `pose_disambiguation` 显式标注。
- **前端依赖**：`computer_code` 需 `yarn install` + `yarn run dev`（或
  `npm run build && npm run preview`，`dist/` 已可构建）；流服务器无需前端也能
  单独拉流。可 `pip install flask flask-socketio flask-cors numpy scipy
  opencv-python ruckig pyserial`（已建 `.venv`）。
- `cv2.sfm`（在线相机标定/BA）pip 版没有；本适配已用闭式回退覆盖其用到的最小
  子集（`fundamentalFromProjections`）。若需完整标定流程，需源码编译 opencv-contrib。
- **通用源的时间戳**：RTSP/USB 为 host-arrival，仅软件对齐；精确动捕仍需
  共同触发/PTP 与硬件时间戳（见 `video_replay.py` 文档）。
