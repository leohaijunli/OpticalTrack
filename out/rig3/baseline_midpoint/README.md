# 底边中点跟踪基线

保存于切换到机体原点跟踪之前：同一组六路 t650 视频，`max_err=6 px²`，
Kalman 9 状态（位置/速度/加速度）、`statePre` 输出、速度测量上限 10 m/s。

这些统计的主位置参考点是底边标记中点。与新算法比较时，使用
`tools/compare_reference_tracking.py` 将旧估计用其测得姿态转换到机体原点，
再对共同有效帧、同一机体原点真值计分。

历史误差视频：`out/video/fused_tracking_error_midpoint.mp4`。
