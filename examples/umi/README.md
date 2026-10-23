# 58 维数据校验与离线评估

这里评估当前遥操方案：双臂绝对 EEF + 双手各 20 维，`state[58]`、预先组成的 `action[50,58]`、三路 RGB 视频。目录名保留 `umi`，输入仍是当前约定的**绝对位姿**数据。原始数据转换见 [遥操数据准备](../teleop_58d/README.md)。

## 1. 校验数据

在项目根目录、项目 Python 环境中执行：

```bash
uv run examples/umi/validate_dataset.py \
  --dataset-dir /mnt/data/lerobot/dzq/teleop_58d_example \
  --report /mnt/data/reports/teleop58_validation.json
```

检查内容：

- LeRobot v2.1 元数据、58 维布局、50 步动作段、任务及 episode 索引。
- 数值有限性、Rotation6D 两列单位长度/正交性、绝对→相对→绝对位姿往返，以及手部通道不变。
- 有效标记与 padding 一致性；存在准备 manifest 时，核对动作偏移和 episode 尾部 padding。
- 三路视频存在、帧数/分辨率/fps 与数据一致，逐帧时间戳对应 `frame_index / fps`。

通过时返回 JSON 报告；失败时返回非零退出码。`--skip-videos` 仅跳过视频探测，仍检查文件存在，并在报告中说明跳过的范围。需要 `ffprobe`。报告文件应放在数据集目录之外，不覆盖已有文件。

校验不能从数值推断旋转列/行的语义、坐标系、标定或手部单位；这些必须与采集程序约定一致。

## 2. 训练后评估 checkpoint

**当前 `LeRobotTeleopDataConfig` 的 policy adapter 仍待接入。** 此命令已经实现，接入与训练相同的 adapter 并得到 58 维 JAX checkpoint 后使用；现在运行会在加载模型参数前明确报出缺少 adapter。

```bash
uv run examples/umi/evaluate.py \
  --dataset-dir /mnt/data/lerobot/dzq/teleop_58d_example \
  --checkpoint /path/to/checkpoint \
  --config-name pi05_teleop_58d \
  --repo-id dzq/teleop_58d_example \
  --prediction-space absolute \
  --max-samples 100 \
  --seed 0 \
  --num-steps 10 \
  --output-dir /mnt/data/reports/teleop58_eval
```

checkpoint 必须包含 `params/` 和 `assets/<asset_id>/norm_stats.json`。评估复用 checkpoint 保存的 58 维归一化统计，不重新统计评估数据。模型只接收当前图像、实测状态和任务提示，不接收真实未来动作。

`--prediction-space` 描述 **`policy.infer()` 完成输出变换后的动作**，而不是模型训练目标的表示：

- `absolute`：输出已恢复为绝对 EEF 位姿。
- `relative`：输出已反归一化，但仍是 `inv(T_obs) @ T_target`。评估按每只手臂当前观测还原整段 50 步位姿，手部数值不变。

后续 adapter 已经还原绝对位姿时，必须选择 `absolute`，避免再转换一次。

默认在完整有效行中均匀选择最多 100 行；`--max-samples 0` 评估全部。`--stride 5` 在每个 episode 的有效行列表中每隔 5 行取样。`--episodes 8 9` 限定 episode；应在训练前按 episode 留出验证集，脚本不会自动保证训练/验证隔离。噪声由 seed 和原始 episode/frame ID 确定。

## 3. 评估已有预测文件

无需模型、JAX 或 GPU，可先验证评估流程：

```bash
uv run examples/umi/evaluate.py \
  --dataset-dir /mnt/data/lerobot/dzq/teleop_58d_example \
  --predictions /path/to/predictions.npz \
  --max-samples 0 \
  --output-dir /mnt/data/reports/teleop58_replay
```

NPZ 必须包含以下字段，不使用 pickle/object 数组：

| 字段 | 形状及含义 |
| --- | --- |
| `episode_index` | 整数 `[N]`，原始 episode ID |
| `frame_index` | 整数 `[N]`，episode 内原始帧 ID，不能是过滤后的行号 |
| `actions` | 有限数值 `[N,50,58]`，**反归一化后的物理量** |
| `action_space` | 标量 Unicode 字符串 `absolute` 或 `relative` |

文件可以乱序，按 ID 匹配目标；重复、未知、无效或带 padding 的样本会报错。采样只在文件已有预测且符合筛选条件的行中进行。脚本自身保存的 `predictions.npz` 可以直接用于此模式，不能再用命令行覆盖其 `action_space`。

## 输出

新建输出目录，包含 `metrics.json` 和 `predictions.npz`：

| 指标 | 含义 |
| --- | --- |
| 左/右 `position_m` | xyz 欧氏距离误差，单位米；均值、P95、距离 RMSE、首尾步及每步误差 |
| 左/右 `rotation_deg` | 两个姿态在 SO(3) 上的夹角，单位度；同上统计 |
| 左/右 `hand_native` | 手部原生单位的 MAE、RMSE、每步及每关节误差 |

同时报告“整个预测段保持当前实测状态”的基线和各 episode 的指标。不把米、角度与手部单位混成一个 58 维 MSE。首次推理耗时单列，包含 JAX 编译；后续耗时包括输入/输出变换。

这是离线动作预测误差，不是机器人闭环成功率。重叠动作段会重复覆盖部分目标时刻，报告按所选动作段等权统计；这里没有置信区间，也不把这些行当作独立实验。
