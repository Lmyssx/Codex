# Step6：递归反射与独立校准约束下的贝叶斯厚度反演

此模块新增于原项目，不修改 Step5 observations、原始 MAT 或历史 Step6。当前真实数据只有接口/质量审计结果；没有独立实测校准和厚度标签，不输出人体厚度或准确度结论。

## 原理与实现

- Blind Reconstruction 的深到浅递归，`eps*=eps'-j eps''`，时间约定 `exp(+jωt)`；末层半无限，前向波 `exp(-jkd)` 衰减。
- IT’IS Skin/Fat/Muscle 四项 Cole–Cole 完整频率曲线，明确 ps/ns/µs/ms；缺参数、未知单位报错，禁止静默补零。
- 脂肪仅有两个跨频率共享的有效介电实部/损耗缩放。它们是有限频带近似，不代表全频因果色散模型。
- 原位置距离门、NumPy FFT 归一化和频率选择严格匹配 Step5。49 个复频点投影为 7 个复/14 个实坐标，观测与预测使用同一 QR 基。
- 发现 96 点循环传播尾部可能绕回；生产配置先将完整测得的系统 CIR 作有限支持零延拓，8 倍时间窗计算组织传播，截取原始 96 个位置后再执行原门控。`linear_convolution_factor=1` 是原 96 点循环近似，不能把增加点数解释为增加观测信息。可用 16/32 倍检查数值误差；未测得的系统窗外响应仍未知。
- 正规化实高斯似然，Cholesky 白化；IG 噪声先验采用 SciPy scale 约定。可建立窗口相关和共享背景误差协方差。
- 粗网格、多起点联合 MAP 与 MCMC 使用同一似然和先验。MCMC 使用坐标 Slice、温度正确的 IG Gibbs、只退火似然的 PT、交换完整状态。
- 可选 `use_map_mixture=true`：冻结 MAP 候选构成的多变量 Student-t 混合提议，每隔数轮作一次带完整 MH/提议密度校正的全局移动。它帮助跨模式，但不改变目标后验；不是论文全部算法的复现。
- 四个独立 ensemble 的冷链用于 rank Rhat、bulk/tail ESS 和 MCSE，s2 也参与验收。热链不纳入厚度区间；不 thinning。

## 安装与运行

在项目根目录安装 `requirements_step6_recursive.txt`。本次验证在 Linux / Python 3.12 上进行，未使用上传的 Windows venv。材料数据库路径在 JSON 中显式配置；本次 Database-V5-0 ZIP 的 ASCII 位于解压项目根，若你的数据库在 Step5 子目录，请只调整 `materials.ascii_path`。

```bash
python -m pip install -r requirements_step6_recursive.txt
python -m step6_recursive_bayes.cli audit --config configs/step6_audit.json
python -m step6_recursive_bayes.cli validate-physics --config configs/step6_audit.json
python -m step6_recursive_bayes.cli estimate-noise --config configs/step6_audit.json
python -m step6_recursive_bayes.cli simulate --config configs/step6_synthetic_fixed.json
python -m step6_recursive_bayes.cli recover-synthetic --config configs/step6_synthetic_fixed.json
python -m step6_recursive_bayes.cli recover-synthetic --config configs/step6_synthetic_uncertain.json
python -m step6_recursive_bayes.cli validate-statistics --config configs/step6_audit.json
python -m step6_recursive_bayes.cli validate-recovery --config configs/step6_synthetic_fixed.json
python -m step6_recursive_bayes.cli validate-coverage --config configs/step6_synthetic_uncertain.json
python -m pytest tests_step6_recursive -q
```

所有结果写入唯一的 `results_step6_recursive/<UTC时间>_<命令>_<唯一ID>/`；用户阅读报告的日期按北京时间说明。每个运行保存配置、种子、输入/源代码 hash 和 Python/依赖/BLAS 信息。`--resume --run-dir <原目录>` 从 checkpoint 恢复；必须使用同一配置、数据、校准与模型。checkpoint 保存 RNG、迭代数、冷链历史和温度状态。`stop_after` 仅用于可复现中断测试，不建议用于研究运行。

`configs/step6_smoke.json` 用于快速通路检查，即便诊断数值看似良好也不会标为收敛验证。

## 真实数据入口与校准

```bash
python -m step6_recursive_bayes.cli map --config configs/step6_real_calibrated.json --record 20260615_171401_171412.mat
python -m step6_recursive_bayes.cli sample --config configs/step6_real_calibrated.json --record 20260615_171401_171412.mat
python -m step6_recursive_bayes.cli report --run-dir results_step6_recursive/某次运行
```

当前前两个命令返回 `calibration_missing`。不会将 S 设为 1，不用空场作 S，也不把未知人体拟合 Psys 当独立校准。

独立校准 NPZ 必须包含 `f_full_hz[96]`（float64）、`system_response_full[1,96]`、`residual_offset_full[1,96]`（complex128）。同名 JSON 提供：`reference_plane, fast_time_origin, fft_iq_convention, background_scheme, hardware_settings_id, reference_ids, repeat_count, method, independent_validation_error, independent_reference, offset_assumption`。真实目标配置 `calibration.target_metadata` 必须独立填写前五项并与校准逐项一致；不得把未核实的硬件元信息填成已验证。

`calibration.fit_reference_response` 对多个独立已知参考的完整复响应拟合加性 L 与乘性 S，至少留一个完整参考对象做验证；若只有门后49点，不能自动恢复完整 S。金属校准需要近似人体加载的已知介电仿体验证迁移。若负载依赖误差明显，需扩展测量模型。

可在独立重复校准给出误差后使用 `residual_delay_sd_s`、`delay_covariance_source`、`delay_basis="center_anchored_group_delay"`。实现的残余延迟基为 `exp(-j2π(f-fc)δτ)`，即中心频率相位已独立锚定的群时延误差；它不等同于额外空气间隙，不同时放开任意相位曲线。未锚定的 RF 时延/相位联合误差需要另建测量模型。

原始 MAT 到位后，audit 会复算所有可用记录；estimate-noise 用同一门/QR 估计收缩协方差。正式真实推断还要求：`noise.path` 指向实测协方差；`validated_repeat_structure=true` 必须有实际窗口稳定性/相关性验证依据；g 的来源必须为独立测量，不能沿用占位值。窗口间漂移和共享背景误差不能被解释为独立重复。

## 输出解释

`d_fat_mm` 为单侧脂肪厚度，`d_skin_mm` 为皮肤厚度，`d_cal_mm=2*(d_skin_mm+d_fat_mm)` 为派生皮褶量，三者分开报告。`xy/zfh` 是姿态，不能推断受试者、部位或真值，也不能把400个窗口视作400人。

- 校准成功只支持条件估计，不等于准确度验证。
- 材料先验宽度和厚度边界是工程起点，必须做敏感性检查。
- 合成覆盖率实验固定皮肤厚度，是独立的条件实验；不声称已完成自由皮肤/材料四维 MCMC 的重复覆盖率验收。
- 预测图的相位展开仅供显示，损失始终来自复数数据的实/虚投影。
- 保留未收敛、积分失败和模式歧义的日志，不用最小残差或单次真值落区间宣称成功。

## 文件职责

`io/types` 数据契约；`operators` 门控/QR；`materials/reflection` 材料与物理；`calibration` 独立参考契约；`likelihood/priors` 概率模型；`inference/samplers` MAP/采样；`diagnostics/reporting` 诊断和输出；`simulate/validation/experiments` 合成与验证；`cli` 命令入口。

## 参考

用户提供的 Blind Reconstruction（式1、Gibbs/PT思想）和 UWB-Fat（仅作厚度/测量原理参考；不能移植其 X7 双通道观测假设）；任务书是实施约束。历史 TMM 仅用于溯源核对，物理参考采用独立的方向一致 TMM。

当前实现不把 AR(1) 噪声相关模型当作任意几何/相位漂移模型。若多窗口相对散布超过预定 0.2 复核阈值，真实入口会阻止共享几何推断；可在不参考厚度标签的前提下配置 `data.observation_indices=[0]` 等预先指定的单窗口，或另行实现并验证显式漂移模型。散布阈值本身是开发 QC 规则，尚未用人体标签调参或验证。
