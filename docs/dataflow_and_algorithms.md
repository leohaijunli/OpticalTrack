# 数据流与核心算法

本文梳理本仓库从数据生成到最终计分的完整链路，以及各环节的核心算法与关键数值。
对应 `README.md` 的「三机 t650」交付。

六路边缘部署、60 Hz 吞吐实测及同步/硬件要求见 [部署评估](edge_architecture.md)。
本页精度是在 30 fps 采样时间轴上完整处理的结果，不代表墙钟实时处理速率。
当前实现的缓存、批量 DLT、刚体向量化与数值回归见 [中心融合优化](fusion_optimization.md)，
完整视频重跑的分位数和分布图见 [优化版误差报告](position_accuracy_optimized.md)。
第二轮计算优化及可选浮点质心试验见 [进一步优化](further_optimization.md)。默认整数质心、
时间关联/滤波规则和本页精度基线均保留。

---

## 一、数据流总览

```
┌────────────────────────────────────────────────────────────────────┐
│ A. 数据生成（离线，Rust + Blender）                                  │
│                                                                    │
│  MAVLink 日志 / UDP  ── ingest ──▶ trajectory(ENU 位置+四元数+速度) │
│                                    └─▶ scenario3.json / .csv        │
│                                                                    │
│  simulate:  6 相机吊装 + 3 机体×4 标记  ──▶ cameras_calib.json      │
│                          rig_blender.json(标记/挂点/内参)           │
│                          detections_2d.csv / markers_3d.csv         │
│                          verify.txt(理想投影自检)                   │
│                                                                    │
│  Blender: 读 scenario + rig  ──▶ 6 路逐帧同步灰度视频               │
│           (CAM01…CAM06, 1280×800, 30fps, 600帧)                    │
└────────────────────────────────────────────────────────────────────┘
                                │  (交付物：6 支视频)
┌────────────────────────────────────────────────────────────────────┐
│ B. 实时链路（Python 适配器，即「数据流」主路径）                      │
│                                                                    │
│ VideoReplay ─▶ 逐帧读 6 路 → (frames, timestamps)                   │
│   │  timestamps = frame / 30  (真实秒)                              │
│   ▼                                                                │
│ prepare_camera_frame ── 灰度→BGR、无畸变、9×9 高斯模糊               │
│   ▼                                                                │
│ detect_visible_dots ── 阈值140 + 连通域 → 整数质心 (面积3..200)      │
│   ▼                                                                │
│ find_point_correspondance_and_object_points                         │
│   ── 跨相机极线匹配 + DLT 三角化 → 3D 点 + 每点重投影 MSE            │
│   ▼                                                                │
│ locate_bodies ── 顶点/底边搜索 + 第4标记 + Kabsch → 刚体位姿         │
│   │   ├─ BodyTracker  (时间关联门 + 姿态先验)                       │
│   │   └─ position_reference = body_origin (pose_t)                  │
│   ▼                                                                │
│ KalmanFilter ── 9状态[p,v,a] → 输出 pos/vel/acc                    │
│   ▼                                                                │
│  emit "object-points" + tracking ─▶ 前端 TrackStatus / 3D 场景      │
└────────────────────────────────────────────────────────────────────┘
                                │  (同一 6 支视频)
┌────────────────────────────────────────────────────────────────────┐
│ C. 计分（只读真值，不参与估计）                                      │
│ evaluate_multi_video ─▶ tracking_from_video.{csv,json}             │
│ accuracy_stats3 ─▶ accuracy3.{csv,json}                            │
│ compare_reference_tracking ─▶ 旧中点 vs 新原点，同参考点配对对比     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心算法（按阶段）

### 1. 数据生成（Rust `mocap-sim` + Blender）

- **轨迹**：MAVLink（`LOCAL_POSITION_NED` 主，`ATT_POS_MOCAP` / `ODOMETRY` 回退）
  → 转换 ENU（x 东 / y 北 / z 上）→ Hermite 插值。位置/速度/加速度连续，
  姿态由四元数表示 + 角速度。
- **相机模型**（`rig.rs`）：针孔无畸变。
  ```
  fx = fy = 2.8mm / 3.0µm = 933.333 px
  cx = (1280−1)/2 = 639.5    cy = (800−1)/2 = 399.5   ← (W−1)/2 像素中心约定
  u = cx + fx·x/z ,   v = cy − fy·y/z
  ```
  主点用 `(W−1)/2` 而非 `W/2`，消除约 −0.5 px 的全局系统偏移。
- **机体几何（4 标记）**：`apex_marker_id` + `base_marker_ids` 显式声明参考三角形
  （底边 → 位置语义，顶点 → 航向），第 4 个**非共面**标记作身份与刚体确认。
  三机边长组合互异（`tarot650_x/fwd/rev`）。
- **Blender 渲染**：同一场景帧号，6 相机同时取同一时刻 → 全局快门同步
  （`max_skew_s = 0`）。EEVEE、灰度、无运动模糊。
  **视频是只需一次生成的交付物**；参考点等算法改动不重新渲染。

### 2. 检测（`helpers.detect_visible_dots`）

```
mask = gray > 140
连通域(8 邻域)，保留 3 ≤ area ≤ 200
质心 = int(质心)   ← 与上游 int(moments) 一致
```

只有标记 LED 过曝（~216），哑光机身/地面最大灰度 78，阈值 140 下电机区不误检。

### 3. 跨相机匹配 + 三角化（`helpers`）

- 各相机 2D 点列表 → 基础矩阵 `F`（`fundamental_from_projections` 闭式回退）→
  极线约束找跨相机对应点对。
- 一对点 → **DLT** 三角化出 3D 点。
- 每点附**重投影 MSE**（px²），供后续拒绝低质量三角解。
- 当前实现按 K/R/t 数值缓存固定几何，批量求解 DLT 与重投影；保留 5 px 极线门限、
  原候选次序与最小误差首项胜出规则。

### 4. 刚体定位（`multi_body.locate_bodies`）——核心

每个机体模板独立跑一次，但共享点所有权（先匹配者占用，防两机共用同一标记）。

**(a) 顶点/底边搜索**（`_locate_one`，保留上游遍历顺序）：
```
候选顶点 i：与 ≥2 个点距离 ∈ [dist1−tol, dist1+tol]
底边对 (a,b)：|a−b| ∈ [dist2−tol, dist2+tol]
pos_测量 = 底边中点；heading = 底边轴方向（π 周期，取负）
error   = 三点重投影 MSE 均值
```

**(b) 过滤**：`max_err ≤ 6`（默认）拒掉坏三角形；时间门检查。

**(c) 几何/姿态**（`_assign_geometry`）：
- 参考三角形 3 点 → **Kabsch** 得 `(R, t)`。
- 用 `R·local_extra + t` 预测第 4 标记位置，在 `extra_tol = 0.05 m` 内找最近
  未占用点；找到 → `pose_mode = rigid`，该点参与最终 Kabsch（4 点 6-DOF）。
- **底边二义消解**：等腰三角形交换底边两点 = 绕中线旋转 180°。第 4 标记找到
  则唯一确定；被完全遮挡时用**旋转测地距离**选与最近一次接受姿态更近的分支
  （`pose_disambiguation = temporal_prior`），超过 `reacquire_gap_s = 0.5 s`
  失效。冷启动/长间断后的三点解记为无解。

**(d) 参考点输出**：
```
position_reference = body_origin：
  pos ← pose_t（刚体拟合原点，即模型名义 CG，位于下中心板中心）
  时间关联门 与 Kalman 都在这个原点工作
  base_midpoint_measured 保留底边中点的未滤波诊断值
```

### 5. 时间关联（`BodyTracker`）

- 输入真实秒级时间戳 `timestamp`。
- **常速度预测**：`predicted = pos_last + vel·Δt`；候选位置与预测距离 > `gate`（默认 0.5 m）则拒。
- `would_accept` 只检查、不改状态；`accept` 才提交；`finish_frame` 每帧最多计一次失锁。
- `rotation_prior` 从最近接受姿态提供 6-DOF 先验，供第 4 标记回退时消歧。
- 时间倒退/循环 → 重置为新时间纪元。

### 6. Kalman 滤波（`KalmanFilter`）

- **9 状态**：`[px, py, pz, vx, vy, vz, ax, ay, az]`（位置/速度/加速度）。
- 转移矩阵含 `dt`（当前帧真实时间差）：位置 += 速度·dt + ½·加速度·dt²；
  速度 += 加速度·dt；加速度为随机游走。
- 测量为位置 + 速度（6 维）；输出取 `correct()` 之后的状态（即**后验**）。
  代码显式读 `statePost`：`cv2 5.0` 的 `correct()` 会就地覆盖 `statePre`，
  旧代码读 `statePre` 得到的其实是同一个后验向量，见[第三轮 §6](spike_robustness.md)。
- 速度测量幅值上限 `max_speed = 10 m/s`（可配置，**不是**创新统计检验）。
- 短时丢帧后（`reacquire_gap_s = 0.5 s` 以内）按丢帧数膨胀位置/速度协方差
  （`reacquire_inflation = 5`，单位是测量噪声 R）；`0` 关闭。见
  [第三轮](spike_robustness.md)。
- 速度另经上游 Butterworth 低通；姿态/身份证据原样透传。
- 标量加速真值来自「同一参考点的二阶后向差分」。

### 7. 计分（独立真值，不参与估计）

- 位置参考点、速度/加速度真值统一取**机体原点**（`out/scenario3.csv` 的
  `x,y,z` + 四元数；原点本身即 `x,y,z`，速度/加速度由一/二阶差分）。
- `accuracy_stats3` 只对实际产生的样本计分，空隙不补；姿态按
  `pose_mode`（rigid / triangle3）分组。
- 航向用 180° 周期最小角距（避免 +45° 与 −45° 两条垂直轴被误判为同一方向），
  并应用仓库负号约定。

---

## 三、关键实现选择

| 项 | 选择 | 理由 |
|---|---|---|
| 检测 | 阈值 140 + 9×9 模糊 + 整数质心 | 与上游 `int(moments)` 一致，标记过曝孤立 |
| 主点 | `(W−1)/2` | OpenCV/渲染像素中心约定，消除 −0.5 px 偏移 |
| 参考点 | 机体原点 `(0,0,0)` | 急转时偏置的底边中点含旋转耦合；原点更贴近整机运动 |
| 二义消解 | 第 4 标记 + 旋转测地距离时间先验 | 等腰三角形底边交换 = 180° 旋转，三点不可区分 |
| 状态 | 9 状态 `[p,v,a]` | 加速度有建模；输出 `correct()` 后的后验（显式读 `statePost`） |
| 丢帧安全 | `rigidity_tol = 2 mm` + `reacquire_inflation = 5` | 形状门拒畸变三角形；膨胀避免重捕获时被旧速度拖住 |
| 过滤 | `max_err ≤ 6 px²` | 拒低质量三角形，raw 最大误差大幅下降 |
| 并发取流 | `_capture_lock` 节流共享 | 多客户端并发 `VideoCapture.read` 会触发 FFmpeg 崩溃 |
| 数据格式 | 原生 int 索引 + `_jsonable` | 服务器 emit JSON，避免 `np.int64` 序列化崩溃 |

---

## 四、当前数值（机体原点，600 帧，实际视频像素）

| 机体 | 覆盖 | KF 位置均值 / P90 / 最大 (mm) | 速率误差 MAE / RMSE (m/s) | 姿态旋转 (rigid, °) |
|---|---|---|---|---|
| T650-A | 98.3% | 3.51 / 4.08 / 13.11 | 0.036 / 0.057 | 0.212 |
| T650-B | 100% | 3.37 / 4.41 / 8.71 | 0.043 / 0.069 | 0.383 |
| T650-C | 96.5% | 4.30 / 4.89 / 17.01 | 0.041 / 0.065 | 0.361 |

同参考点、共同有效帧配对对比（旧中点滤波估计转换到同一原点后）：
KF 最大误差 A 89.65→13.11、B 53.48→8.71、C 32.57→17.01 mm。
详见 `out/rig3/reference_comparison.json` 与 `error_source_audit.md`。
