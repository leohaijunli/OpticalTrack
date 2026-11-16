# mocap-sim

MAVLink 驱动的光学动作捕捉仿真：支持单架 Crazyflie 2.1 和三架 Tarot Ironman 650，
在机库内的 6 机位捕捉体积里飞行，输出 6 路严格同步的单色
（OV9281）相机画面、2D 检测真值和 3D 位姿真值，可直接作为开源光学定位/跟踪
算法的参考数据。

> 📘 **数据流与核心算法**：[docs/dataflow_and_algorithms.md](docs/dataflow_and_algorithms.md)
> 梳理从数据生成到最终计分的完整链路，以及检测、跨相机三角化、刚体定位、
> 时间关联、Kalman 滤波等各环节算法与关键数值。
>
> 🖧 **分布式边缘检测架构**：[docs/edge_architecture.md](docs/edge_architecture.md)
> 六路完整 600 帧切分等价验证、60 Hz 处理预算、单网线双流与曝光同步。
> 当前 i7-1255U 中心融合平均 105.70 ms/组，尚不满足 60 Hz；详见实测与优化优先级。

```
MAVLink 日志 / UDP  ──►  mocap-sim  ──►  scenario.json + rig/*.csv
                              │                    │
                              │                    └──► Blender (6 机位吊装, 可见光)
                              └──► 三角化自检 (重投影 RMS / 6-DOF 误差)
```

## 1. 快速开始

### 三机视频与误差曲线（当前交付）

- **完整视频（机体中心参考点）**：[六机位 + 三机误差曲线 + 真值/Kalman 轨迹](out/video/fused_tracking_error_body_origin.mp4)
  （2600×900，30 fps，600 帧，20 秒）。摄像头面板只显示当前检测，不叠加历史轨迹。
- 干净视频：`out/video/01_cam01_t650_20s.mp4` … `06_cam06_t650_20s.mp4`；
  带编号/时间的版本为同名 `_hud.mp4`。
- 实测 CSV：[tracking_from_video.csv](out/rig3/tracking_from_video.csv)；
  配置及逐机统计（含视频/标定/算法哈希溯源）：[tracking_from_video_summary.json](out/rig3/tracking_from_video_summary.json)。
- 定位、测速与姿态统计：[accuracy3.json](out/rig3/accuracy3.json)；
  误差来源：[error_source_audit.md](out/rig3/error_source_audit.md)。
- 三架机的亮斑均经实体杆连接到中心板，安装坐标为 `out/rig3/rig_blender.json` 中的 `mount`。

算法基线是 Low-Cost-Mocap 的 5 px 极线候选搜索、DLT 和 Kalman `statePre` 输出；
匹配器保持上游的顶点/底边遍历顺序，只是把"等腰三标记"推广为**每机 4 标记**的刚体
签名：前 3 个组成参考三角形（保留上游顶点/底边搜索与基边轴航向），第 4 个为非共面标记，
用于身份及底边顺序确认，4 个匹配点共同参与 Kabsch 6-DOF 拟合。
可见光检测仍采用同一套 9×9 模糊和亮度阈值。

主跟踪参考点为**机体坐标原点 `(0,0,0)`**，当前 Tarot 模型位于下中心板中心。
它是固定的几何参考点（模型名义 CG），没有通过质量分布计算或实测真实质心。
四标记 Kabsch 的 `pose_t` 作为原始位置进入时间关联门和 9 状态 Kalman；
`pos/vel/acc` 都对应这个原点。`base_midpoint_measured` 保留底边中点的原始测量。
姿态仍另给出 `pose_r/pose_t/pose_euler_zyx_deg/pose_rmse_m`，其中 `pose_t` 是未滤波位姿。
真值位置、速度、加速度统一按该原点计分；旧中点基线保存在
`out/rig3/baseline_midpoint/`，同参考点对比见 [reference_comparison.json](out/rig3/reference_comparison.json)。

**当前布局的取舍**：三个参考三角形的尺寸不同，但等腰三角形仍有底边两点交换的
180° 姿态二义性。第 4 个非共面、偏离对称面的标记按预测位置匹配（默认 0.05 m
容差），提供几何确认（`pose_mode=rigid`）。第 4 标记被遮挡时使用三点拟合
（`pose_mode=triangle3`），并选择与最近一次接受姿态旋转距离最小的分支。
先验超过 0.5 s 失效；冷启动或长间断后的三点解仍可能二义，原点模式将其记为无解，
等待新的几何确认，防止把不确定的原点混入跟踪。
`pose_disambiguation` 明确标记 `extra_marker`、`temporal_prior` 或 `unresolved`。
已知点身份的三个非共线点足以定刚体姿态，这里的问题是对称标记的身份顺序。
视频保留失锁空隙和尖峰，没有用真值校正估计。
实际视频的位置误差不能用 `verify.txt` 或理想投影 CSV 的亚毫米自检值替代。

复现当前视频：

```bat
.venv\Scripts\python.exe tools\evaluate_multi_video.py --position-reference body_origin --max-err 6
.venv\Scripts\python.exe tools\accuracy_stats3.py
.venv\Scripts\python.exe tools\make_multi_curve_video.py --out out\video\fused_tracking_error_body_origin.mp4
.venv\Scripts\python.exe tools\compare_reference_tracking.py
.venv\Scripts\python.exe -m unittest tests.test_mocap_adapter tests.test_video_evaluation
```

与视频源、真值解耦的分析接口在 `adapters/low_cost_mocap/tracking_pipeline.py`。
`adapters/low_cost_mocap/run_tracking.py` 可接视频、图片序列、RTSP/MJPEG 或 USB 设备；
真实相机需要自己的内外参及硬件同步，通用 OpenCV 到达时间不等于曝光时间。
应用启动与真实源配置见 [适配器说明](adapters/low_cost_mocap/README.md)。

### 原单机数据集

```bat
cargo build --release --manifest-path rust\mocap-sim\Cargo.toml

:: 生成演示遥测（MAVLink 2，8 种消息，混合速率）
rust\mocap-sim\target\release\mocap-sim.exe demo

:: 解析日志 -> 轨迹 -> 场景
rust\mocap-sim\target\release\mocap-sim.exe ingest --file logs\demo_crazyflie.mavlink --fps 30 --duration 20

:: 相机机位、2D 检测、三角化自检
rust\mocap-sim\target\release\mocap-sim.exe simulate
```

产物：

| 文件 | 内容 |
| --- | --- |
| `out/scenario.json` | 每帧位姿（ENU/Z-up，米）、速度与角速度，供 Blender 使用 |
| `out/rig/rig_blender.json` | 6 台相机位姿/OV9281 内参、3 个标记的机体系坐标 |
| `out/rig/cameras_pose.json` | 纯相机位姿（无 Blender 依赖） |
| `out/rig/cameras_calib.json` | OpenCV 风格标定：`K`、`dist`、`R`、`t`、`fx/fy/cx/cy`、传感器/镜头元数据、时间与同步说明 |
| `out/rig/detections_2d.csv` | 每帧每相机每个标记的像素坐标、半径、亮度 |
| `out/rig/markers_3d.csv` | 标记世界坐标真值 + 可见相机数 |
| `out/rig/trajectory.csv` | 位姿/速度/角速度真值 |
| `out/rig/verify.txt` | 重投影 RMS、6-DOF 误差、可见性统计 |

## 2. 接入 MAVLink

两种输入方式，解析路径完全相同：

```bat
:: A. 已有日志 —— .tlog（QGC，带 8 字节时间戳）或裸 .mavlink 流
mocap-sim ingest --file recording.tlog --fps 30

:: B. 直接监听 UDP（例如 SITL / 数传转发的 14550）
mocap-sim ingest --udp 0.0.0.0:14550 --seconds 30 --fps 30
```

`--udp` 会写一份 `logs/captured.mavlink`，随后走同一条解析链路，因此录下来后可
离线重放。`ingest --duration 20` 固定取 20 s；解析出的轨迹再做 Hermite 插值，
位置/速度/加速度连续，姿态与角速度由四元数紧邻差分得到。

### 支持的消息

| 消息 | 用途 |
| --- | --- |
| `LOCAL_POSITION_NED` | 首选位置 + 速度源（NED） |
| `ATT_POS_MOCAP` | 动捕系统回灌的位姿（位置回退源） |
| `ODOMETRY` | 位置/姿态/速度回退源 |
| `ATTITUDE` | 欧拉角姿态（优先级最高） |
| `HIGHRES_IMU` / `GLOBAL_POSITION_INT` | 一致性、录制完整性 |
| `SET_POSITION_TARGET_LOCAL_NED` | 离线 setpoint（速度 + yaw_rate） |

位置源按 `LOCAL_POSITION_NED → ATT_POS_MOCAP → ODOMETRY` 依次回退，姿态源按
`ATTITUDE → ATT_POS_MOCAP → ODOMETRY → 动力学推导` 回退；实际采用哪一路会打印
并写入 `scenario.json` 的 `meta.position_source` / `meta.attitude_source`。

### 协议实现

方言不是硬编码的：`rust/mocap-sim/data/*.xml` 直接取自 MAVLink 官方
`message_definitions/v1.0`，程序从中推导

* 字段 wire 顺序（按元素宽度降序稳定排序，extensions 追加在末尾）
* `CRC_EXTRA`（消息名 + `type name`，数组字段追加元素个数原始字节）
* MAVLink 2 payload 截断规则（CRC 基于截断后的实际 payload）

`cargo test` 内含 25 个规范 CRC_EXTRA 值以及 v2 编解码往返测试；生成的日志用
pymavlink 交叉验证为 0 BAD_DATA。

### 坐标约定

* MAVLink / PX4：NED 世界、FRD 机体
* Blender / 数据集：ENU 世界（x 东、y 北、z 上，米）
* 转换集中在 `src/flight.rs`（`attitude_to_blender` / `blender_to_attitude_frd`）

## 3. 相机与场景

### Arducam OV9281 + M27280M07S

| 参数 | 值 |
| --- | --- |
| 传感器 | OV9281，1/4"，单色，global shutter |
| 分辨率 | 1280 × 800 |
| 像元 | 3.0 µm × 3.0 µm |
| 镜头 | M27280M07S，M12，EFL 2.8 mm，F/2.8 |
| 内参 | `fx = fy = 2.8 mm / 3.0 µm = 933.333 px`，`cx = 639.5`，`cy = 399.5`（整数像素中心约定） |
| 视场 | HFOV 68.88°、VFOV 46.40°（按无畸变针孔计算） |
| 畸变 | `k1 = k2 = 0`；镜头标称 75° HFOV 含桶形畸变，本数据集不建模，保证渲染像素与投影真值严格一致 |

场景中 Blender 相机的物理量来自真值 JSON：`lens = 2.8 mm`、
`sensor_width = 3.84 mm`（= 1280 × 3 µm），输出 1280 × 800。

### 6 机位吊装布局（紧凑 5 m 版本）

* 环形半径 3.5 m、安装高度 4.8 m（**≤ 5 m 硬约束**，由 `layout.rs` 断言），
  方位 30° / 90° / 150° / 210° / 270° / 330°
* **标准编号**：俯视逆时针编号，自 +X（东）起算，间隔 60°——
  `CAM01 = 30°`、`CAM02 = 90°`、`CAM03 = 150°`、`CAM04 = 210°`、`CAM05 = 270°`、
  `CAM06 = 330°`。编号、`cameras_calib.json` 数组顺序、`detections_2d.csv` 的
  `cam_name` 与交付视频文件名 `01_cam01` … `06_cam06` 完全一致
* 全部通过吊杆/吊架安装在机库屋顶结构上，**没有任何落地立柱或地面三脚架**
* 每台相机略偏向体积中心外侧瞄准，避免六台相机主光轴共点导致单目退化
* 机库 14 × 12 × 6.2 m（屋顶桁架 5.9 m，相机导轨 5.25 m，留吊装净空）：
  哑光混凝土地面（切缝分格）、波纹钢墙、屋架/檩条、推拉机库门、4 盏吸顶灯、
  静态标定架与标定杆。墙面/地面反照率压到 0.03–0.09，只有标记 LED 过曝，
  因此 140 阈值下不会和地面连成大片亮区
* 捕捉体积 3.8 × 2.6 × 2.7 m，中心 (0, 0, 1.35)，起飞点 (0, −0.65, 0.30)
* 标记：每机 **4 个直径 24 mm** 的 LED 球，安装在碳杆上，杆脚落在中心板或
  机臂实体上（`rig_blender.json` 的 `mount` 坐标）；电机/桨座均为哑光深色，
  审计中电机区最大灰度 75（阈值 140）

三架机的参考三角形互不相同（`tarot650_x / tarot650_fwd / tarot650_rev`）：

| 机体 | 底边 (m) | 腰 (m) | 第 4 标记 |
| --- | --- | --- | --- |
| T650-A `tarot650_x` | 0.460 | 0.569 | 中心板上方，偏离对称面 |
| T650-B `tarot650_fwd` | 0.280 | 0.340 | 尾部偏左 |
| T650-C `tarot650_rev` | 0.340 | 0.444 | 尾撑上方，偏右 |

第 4 标记不共面，三角形边长组合也互不相同；完整几何提供单帧身份确认，
多帧叠加 0.5 m 常速度预测门限，并在三点回退时做姿态分支的时间消歧。

相机安装高度高于捕捉体积，俯角约 46°，画面上边缘仍在地平线以下 23°，因此
吊架、屋架、灯具都不会进入任何交付画面；`out/rig3/verify.txt` 与
`blender/debug_ray.py` 可用于复核（对画面多点 raycast，全部命中地面）。

### 3D 布局图（可旋转 / 缩放）

```bat
python tools\make_rig_viz.py      :: -> out\viz\camera_rig_3d.html
```

输出是一个**完全离线、单文件**的 HTML（无 CDN、无依赖），双击即可打开：

* 机库外壳（墙 / 屋顶 / 桁架 / 推拉门 / 灯具）与 14 × 12 m 地坪
* 6 台相机的真实视锥（由各自的 OV9281 内参 + 位姿算出）、吊杆与屋顶吊点
* 右侧面板可独立显示 / 隐藏 `CAM01`…`CAM06` 的视野；图例中的同名开关同步更新
* 编号徽章 1–6、地面方位环（30°…330°）与逆时针编号箭头
* 三架机的飞行轨迹（A/B/C 三色）、捕捉体积、起飞垫、坐标轴与 5 m 比例尺

操作：拖拽旋转、滚轮缩放、右键拖拽平移、双击相机聚焦；`R` 复位、
`H` 机库全景、`T` 俯视、`S` 侧视。

视野控制：右侧「相机视野（视锥）」中的六个勾选框逐机位控制显示；
「全选 / 清空」可批量切换。取消「显示已选视野」会暂时隐藏全部视野并保留
单机选择，重新勾选即可恢复。仅隐藏视锥，不隐藏相机本体、编号或吊架；
旋转、缩放、切换视角和复位均保留当前选择。小窗口中图例隐藏时，右侧开关仍可操作。

### 帧同步

六个交付视频来自**同一个 Blender 场景、同一条三机轨迹**（`out/scenario3.json`，
由 MAVLink 日志生成）。渲染时逐帧切换 scene frame，六个机位在同一帧号下取的是
同一个时刻的同一批物体：

* 帧号 0 基：`f0000.png` … `f0599.png`，`t = frame / 30 s`
* `detections_2d.csv`、`markers_3d.csv` 与 PNG 帧号一一对应；位姿/速度真值为
  `out/scenario3.csv`（每帧 3 行，`body_id` 区分）
* 全局快门（OV9281），同一帧内四个标记同时曝光；`cameras_calib.json` 的
  `time` / `sync` 段显式记录 `max_skew_s = 0`
* 编码时 6 路视频统一 `-framerate 30 -start_number 0`，不做任何抽帧/补帧
* 编码前检查六份渲染清单均完成、各 600 帧且源哈希一致；
  `out/video/t650_manifest.json` 记录成片哈希，评估缓存绑定视频、标定与算法版本

## 4. 渲染

```bat
set BLENDER=vendor\blender-4.5.16-windows-x64\blender.exe
set MOCAP_SCENARIO=scenario3.json
set MOCAP_RIG=out/rig3/rig_blender.json

:: 单帧预览（帧号 0 基；抽帧用 --frame-list）
%BLENDER% -b --factory-startup --python blender\mocap_scene.py -- ^
  --mode still --shots CAM01 --band vis --tag t650 --samples 8 --frame-list 0,150,300

:: 6 机位 × 600 帧成片
%BLENDER% -b --factory-startup --python blender\mocap_scene.py -- ^
  --mode render --shots CAM01,CAM02,CAM03,CAM04,CAM05,CAM06 --band vis ^
  --tag t650 --width 1280 --height 800 --samples 8 --frames 0-599
```

* 相机 POV 直接输出 8-bit 灰度 PNG（`color_mode = BW`），匹配 OV9281 mono；
  hero/closeup 文档镜头仍为 RGB。
* EEVEE Next，8 samples，关闭运动模糊（global shutter + 利于质心检测）。
* `--band vis` 为可见光；`--band ir` 保留为 850 nm 诊断模式，交付片不使用。
* `python tools\encode_multi.py t650` 生成 6 支干净版和 6 支 HUD 版：

```
out/video/01_cam01_t650_20s.mp4        out/video/01_cam01_t650_20s_hud.mp4
...
out/video/06_cam06_t650_20s.mp4        out/video/06_cam06_t650_20s_hud.mp4
```

HUD 版左下角为机型/标记说明，右上角为 `frame` 与 `t`（与 CSV 帧号一致）。

## 5. 自检数据

### 几何自检（`out/rig3/verify.txt`，理想投影）

```
detections        : 41914
triangulation rms : 0.0377 px (mean over 7200 marker-frames)
6DOF position err : 0.163 mm mean
6DOF rotation err : 0.037 deg mean
```

即：把真值投影得到的 2D 检测重新三角化，可还原到亚毫米/亚度级真值；理想视场
检查中每个标记均落在至少两台相机内。实体遮挡及像素检测缺失由实际视频另行计分。
24 mm 标记在 5–7 m 处直径约 3–5 px，
与真实 OV9281 + 2.8 mm 镜头的成像尺度一致。

### 实际视频精度（机体原点，`out/rig3/accuracy3.json`，MP4 像素）

| 机体 | 覆盖 | 位置误差 (mm) | 测速 MAE/RMSE (m/s) | 航向 (°) | 姿态（Kabsch） |
|---|---|---|---|---|---|
| T650-A | 98.3% | 3.51±0.84（max 13.11） | 0.036 / 0.057 | 2.5±6.6 | 原点 3.47 mm，角 0.21° |
| T650-B | 100% | 3.37±0.86（max 8.71） | 0.043 / 0.069 | 2.3±2.5 | 原点 3.40 mm，角 0.38° |
| T650-C | 96.5% | 4.30±1.55（max 17.01） | 0.041 / 0.065 | 2.2±3.2 | 原点 4.28 mm，角 0.36° |

* `--max-err=6` 拒绝低质量参考三角形；原点 raw 最大误差 A/B/C 为 6.25/6.01/20.69 mm。
* **同一参考点对比**：旧中点滤波输出用其测得的 `R` 与标记安装偏移转换到机体原点，
  再与新结果在共同有效帧上计分。KF 最大误差 A 89.65→13.11、B 53.48→8.71、
  C 32.57→17.01 mm；覆盖帧数均保持 590/600/579。
* 新滤波器仍采用上游 `statePre`，并保留可配置的 10 m/s 速度**测量幅值**上限。
  该上限不是速度创新统计检验。改变参考点减轻了急转时偏置标记中点的运动耦合；
  剩余噪声、遮挡和预测滞后仍存在，不代表尖峰必须靠改变参考点才能改善。
* 加速度来自 9 状态模型的预测状态；相对原点位置二阶后向差分，矢量误差 RMSE
  A/B/C 为 1.18/1.59/1.17 m/s²，详见 `accuracy3.csv/json`。
* 第 4 标记偶发被遮挡时退回三点解：本轮 5 帧由时间先验正确选支
  （`pose_disambiguation=temporal_prior`），无 180° 姿态翻转
* 图像侧抽帧审计（`out/rig3/led_contrast_4marker.json`，帧 0/150/300/450/599）：
  359/360 标记命中、0 误检亮斑、电机区最大灰度 78（阈值 140）
* 帧同步复核（`validate_frames.py`）：347/347 检测命中，质心平均偏差 0.104 px
* 误差来源与修复过程：[error_source_audit.md](out/rig3/error_source_audit.md)

渲染完成后可再用图像本身复核一遍（同时验证帧对齐与标记对比度）：

```bat
.venv\Scripts\python.exe tools\validate_frames.py --frames 0,150,300,450,599
.venv\Scripts\python.exe tools\validate_marker_contrast.py --pattern "{cam}_t650" ^
       --frames 0,150,300,450,599 --cams CAM01,CAM02,CAM03,CAM04,CAM05,CAM06 ^
       --out out\rig3\led_contrast_4marker.json
```

`validate_marker_contrast.py` 用真实检测器统计命中标记数、误检亮斑、电机区灰度，
真值只用于事后一对一计分，不参与检测。

脚本在每张 PNG 里提取饱和亮斑质心，与 `detections_2d.csv` 的预测像素匹配；
运动明显的帧可帮助发现错帧；严格同步还由共同场景帧号、渲染清单和编码帧数共同核验。

## 6. 目录

```
rust/mocap-sim/       MAVLink 解析、轨迹、相机投影、2D 检测、三角化校验
blender/mocap_scene.py 机库场景、6 机位吊装、机体动画与渲染
blender/debug_ray.py   相机 POV 遮挡排查（raycast）
tools/make_rig_viz.py  生成可交互 3D 机位/机库布局图
tools/validate_frames.py  渲染帧亮斑 vs 2D 检测（兼验帧同步）
tools/validate_marker_contrast.py  真实检测器的标记命中/误检/电机高光审计
tools/evaluate_multi_video.py  视频或图片序列 → 定位/姿态精度（含真值 Kabsch）
tools/accuracy_stats3.py       定位、测速、航向、6-DOF 姿态统计
tools/compare_reference_tracking.py  将旧中点估计转换到同一原点作配对对比
tools/verify_body_origin_app.py  通过实际 MJPEG/Socket.IO 核验原点、p/v/a 和时间先验
tools/make_multi_curve_video.py 六机位 + 误差曲线 + 轨迹合成视频
tools/encode_multi.py   ffmpeg 编码 6 支 20 s 单色视频（干净 + HUD）
configs/tarot650_5m.json  5 m 紧凑布局参数（唯一空间真值来源）
out/rig3/              标定、检测、真值 CSV/JSON、验证报告
out/viz/               交互式 3D 布局图（camera_rig_3d.html）
out/frames/            逐帧 PNG（CAM01_t650 … CAM06_t650）
out/video/             交付视频
```
