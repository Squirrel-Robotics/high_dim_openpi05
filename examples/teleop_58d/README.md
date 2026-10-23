# 58 维遥操数据准备

运行入口：`scripts/prepare_lerobot_dataset.py`。当前处理**一个原始 episode**，输出本地 LeRobot v2.1 数据集，支持当前仓库固定的 LeRobot 版本。不上传数据、不覆盖已有数据集。

## 数据含义

| 范围 | `observation.state[58]` | `action[50,58]` |
| --- | --- | --- |
| `0:9` | 左臂实测绝对 EEF：xyz + Rotation6D | 左臂下发绝对 EEF 目标 |
| `9:18` | 右臂实测绝对 EEF：xyz + Rotation6D | 右臂下发绝对 EEF 目标 |
| `18:38` | 左手 20 维实测关节值 | 左手 20 维下发命令 |
| `38:58` | 右手 20 维实测关节值 | 右手 20 维下发命令 |

位置单位为米；双臂实测与目标必须使用相同参考坐标系。Rotation6D 按旋转矩阵前两**列**拼接：`[R00,R10,R20,R01,R11,R21]`。手部保留记录的原始单位，不套用旧 6 维手的 0–1000 缩放。

这里保存绝对位姿。后续 policy adapter 在归一化前调用 `umi_pose.absolute_actions_to_relative()`，对每只手臂按当前观测计算整段动作：`R_rel = R_obs.T @ R_target`，`p_rel = R_obs.T @ (p_target - p_obs)`。已经变成相对位姿的 UMI 数据不能作为此入口的绝对 EEF 输入；需先明确原点和变换约定，再接入单独适配。

## 原始输入

提供的 UMI 示例只作为相机和 Wuji 文件格式参考。示例中的 Pico grip 没有被当作机器人 EEF。遥操必须另外提供两个 CSV：

- `--state-pose-csv`：实测 EEF 位姿。
- `--target-pose-csv`：实际下发的 EEF 目标，不能用实测轨迹代替。

两个 CSV 都需要以下表头，每行一条双臂记录：

```csv
host_monotonic_ns,left_px_m,left_py_m,left_pz_m,left_rot6d_0,left_rot6d_1,left_rot6d_2,left_rot6d_3,left_rot6d_4,left_rot6d_5,right_px_m,right_py_m,right_pz_m,right_rot6d_0,right_rot6d_1,right_rot6d_2,right_rot6d_3,right_rot6d_4,right_rot6d_5,valid
```

`valid` 可省略，默认 true。时间为采集主机 `CLOCK_MONOTONIC` 纳秒，严格递增。不能直接填墙钟、机器人设备时钟，或来自另一台主机的 monotonic 时间；要先转换到本次采集主机时钟。实际遥操的字段命名可以在 `read_pose_csv()` 中映射，数据语义保持不变。

`--raw-dir` 下使用以下已核对的文件布局：

```text
cameras/head_cam/rgb/{frames.csv,aligned_frames.csv,clock_model.json,part-000001.mp4}
cameras/left/rgb/{frames.csv,left_cam.mp4}
cameras/right/rgb/{frames.csv,right_cam.mp4}
hands/wuji/left/{metadata.json,joint_states.jsonl,commands.jsonl}
hands/wuji/right/{metadata.json,joint_states.jsonl,commands.jsonl}
```

头部视频也可为 `part-000001.h265`；当前支持单 part，完整保留头部画面，不自动裁剪双目。视频宽度默认 640，高度按比例调整到偶数。

对齐规则：

- 头部使用 `aligned_frames.csv` 的 `mapped_5b_monotonic_ns`，要求时钟映射成功；手腕使用 `source_timestamp`，要求 `v4l2_monotonic`。
- 手部反馈使用 `host_receive_mono_ns`；命令使用主机时钟下的 `accepted_mono_ns`，并校验接受状态与记录时间。
- 手反馈按 NID 重排：`1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24`。命令没有 NID 时，要求相邻 metadata 声明 `joint_order=thumb_index_middle_ring_pinky_S1_to_S4`。
- 在所有流的公共区间生成默认 10 Hz 网格。位姿/手部取当前时刻及之前最近一条，默认最多允许 50 ms 陈旧；相机取最近帧，默认偏差不超过 40 ms。
- `--command-field position` 默认读取记录的 `position`。若采集程序确认实际下发量是 `target_position`，显式指定该选项；两字段不混用。
- 默认动作对应 `t+0.1s` 到 `t+5.0s`。如果采集约定需要包含当前时刻，使用 `--action-offset 0`。两种约定不得混入同一训练集。
- 保留全部时间网格行；不足 50 步的尾部重复最后目标并设置 `action_is_pad`。完整动作、当前状态和相机均有效时，`sample_valid_h50=1`。现有 dataloader 会过滤无效行。

## 运行

在能访问原始数据的机器上，使用本项目环境（包含固定版本的 `datasets`），并确保 `ffmpeg`、`ffprobe` 在 PATH：

```bash
uv run scripts/prepare_lerobot_dataset.py \
  --raw-dir /path/to/teleop_episode \
  --state-pose-csv /path/to/measured_eef.csv \
  --target-pose-csv /path/to/commanded_eef.csv \
  --repo-id dzq/teleop_58d_example \
  --task "Put the object into the box"
```

默认输出到 `$HF_LEROBOT_HOME/dzq/teleop_58d_example`；未设置时使用 `$HF_HOME/lerobot`，通常是 `~/.cache/huggingface/lerobot`。自定义位置示例：

```bash
export HF_LEROBOT_HOME=/mnt/data/lerobot
# 在同一环境中执行准备脚本和后续训练。
```

也可单独传 `--output-root /mnt/data/lerobot`，训练时仍需设置匹配的 `HF_LEROBOT_HOME`。输出包含 Parquet、三路重采样视频、任务/episode/统计元数据和 `meta/prepare_manifest.json`；manifest 记录来源、维度约定、时间阈值与有效样本数。

该步骤只准备数据。当前 `LeRobotTeleopDataConfig` 的 policy adapter 仍待接入，因此暂不能直接开始训练。接入 adapter 后再计算 OpenPI 的归一化统计；此处 LeRobot 元数据中的统计不能代替它。单条 episode 仅用于检查流程，训练前需汇总更多 episode 并按 episode 划分训练/验证集。
