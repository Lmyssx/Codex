# Step6 实施与验证报告

日期：2026-10-25（北京时间）。所有人体厚度输出保持禁止状态；本报告中的厚度数值均为合成实验。

## 实际项目与实施范围

已实际读取 Step5 README_W110.md、build_observations.py、全部40个 observations、IT’IS ASCII/历史转换器和 TMM，并核对两篇论文。新增 step6_recursive_bayes、configs、tests_step6_recursive；未修改原始输入。原始 dataset.zip 仍因32 MiB导入限制而未取得，不能把任务书中的原始MAT复算结论当作本次验证结果。

实际 observations 路径为 Step5_observations/observations/{xy,zfh}-observations；README示例目录与实际不同。README声称中心频率±1.50 GHz，代码与保存轴实际为±0.75 GHz。构建脚本默认在 Step5 下找 dataset，新配置明确使用项目根 dataset。

## A：数据契约与只读审计

- 40记录、400窗口；每个 H_obs=(10,1,49)，complex128，有限。频率、W=110、stride=110、NFFT=96、原门1..7全部通过契约核对。
- QR秩为7复/14实；最大子空间重建相对误差 5.7688e-16。
- 记录内相对窗口散布为 0.3111–0.9984，全部触发0.2开发复核阈值；这不是帧内相干性估计。真实入口阻止对明显漂移的多窗口直接共享几何，支持预先指定单窗口。
- 1190个已导入的原文件与原ZIP/单文件附件逐字节一致（不计编译缓存）；40个 observations 未被改写。
- 原始MAT重建、帧内相干性、实测空场协方差待原始文件到位。用生成MAT核对原Step5函数与新算子通过，不能替代原始实测复算。

## B：材料、递归、TMM与门控

- 采用论文式(1)的深到浅递归；独立方向一致TMM与递归最大复数差 3.5451e-16。随机多个层数、被动性、零厚度、强损耗、已知延迟和算子测试通过。
- 皮肤1.5 mm/脂肪15 mm/g=0：递归幅值0.77277–0.79876；历史TMM幅值1.20530–1.53618。历史右到左传播应由diag(P,1/P)改为diag(1/P,P)；修复收益与递归实现差异严格区分。旧文件保持不变。
- IT’IS tau3 原字节0xb5是微符号；旧UTF-8 replacement读法损坏单位。新解析器识别编码，验证ps/ns/µs/ms，精确匹配组织名并拒绝缺失/NaN参数；全网格计算与旧49点CSV吻合。
- 96点循环近似相对32倍长窗基准有约1.29e-4差异，故加入有限系统CIR零延拓的长卷积。默认8倍与32倍差约1.16e-5；显式尾部绕回压力测试通过。仍需对真实系统窗外支持及校准参考面独立验证。
- 不强制复基带共轭对称；门前相乘与错误门后相乘的不等价测试通过。

## C：概率模型、校准接口与恢复

- 实现规范化高斯似然、Cholesky白化、收缩协方差、时间相关/共享背景块及IG噪声尺度。空场不可用时 estimate-noise 明确返回 raw_empty_data_missing，不填造协方差。
- 独立校准要求完整96点、单通道、出处JSON与目标元信息匹配；参考拟合仅使用已知参考，至少留一个完整参考对象验证。合成系统响应带synthetic标志，真实入口拒绝它。
- 无噪声MAP最大厚度误差 1.494e-09 mm。
- 有噪声固定材料：多模式MAP筛查后的密集二维局部积分（241×321，覆盖显著模式±12个局部SD，解析积分s2）与MCMC均值/方差通过比较。df积分均值14.994952 mm，MCMC均值14.995109 mm；方差0.00074041与0.00075552 mm²。

## D：Slice-Gibbs-PT与统计验证

- 坐标Slice采用随机stepping-out预算和shrinkage；只退火似然。IG条件shape=alpha0+D/(2T)、scale=beta0+SSE/(2T)；完整状态交换、缓存更新、checkpoint/RNG恢复通过测试。
- 已知有界均匀/截断正态/双峰分布、线性高斯解析后验、正态均值+IG方差案例通过；增加全局提议后重跑高斯分布验证通过。
- 首轮纯坐标Slice/PT在次优模式停留，Rhat约1.03且ESS不足，明确记为失败，未纳入验收。采用冻结的多模式Student-t提议作MH全局移动，保留Slice/Gibbs/PT；接受率含提议密度比，目标后验不变。
- 正式合成运行均为4个独立ensemble，400 warmup+1400 retained draws；固定材料4温度、材料不确定性6温度，最高温40。长度根据实际诊断验收，不把smoke计为验证。
- 固定材料：最大Rhat=1.001017，最小bulk ESS=5185，最小tail ESS=4565。
- 材料不确定性：最大Rhat=1.001970，最小bulk ESS=1460，最小tail ESS=2037。s2纳入以上诊断。

## E：公平对照、校准误差与覆盖率

- 自由皮肤厚度的同一份数据/同一协方差/同一校准已逐元素核对。生成真值ds=1.5 mm、df=15 mm、u_eps=0.08、u_loss=-0.10。
- 固定材料模型主要模式df≈8.402 mm，支持约98.75%；次模式≈15.386 mm，支持约1.25%。模式占用诊断通过，按多峰单独报告；不能用跨模式均值概括。标称噪声下残差提示失配（描述性p≈1.12e-6），自由s2会吸收部分失配。
- 材料不确定性模型df中位数15.066 mm，95%区间[14.966,15.165] mm；u_eps/u_loss为跨频率共享扰动。该合成例说明材料失配会改变厚度模式，不能外推为人体精度提升。
- 重复实验固定ds=1.5 mm，df与材料/独立中心锚定群时延误差为待推断量。固定模型全局一维密集积分；扰动模型使用多模式厚尾重要性积分，s2解析积分。每组24次全部通过数值积分诊断；与自由皮肤四维MCMC单次验证分开。

| 场景 | 模型 | 覆盖次数 | 覆盖率的Wilson 95%区间 | 平均区间宽度mm |
|---|---|---:|---:|---:|
| 独立校准群时延误差 | 忽略校准误差 | 6/24 | [0.120, 0.449] | 4.724 |
| 独立校准群时延误差 | 传播校准误差 | 22/24 | [0.742, 0.977] | 0.130 |
| 材料变化 | 固定材料 | 8/24 | [0.180, 0.533] | 0.533 |
| 材料变化 | 材料不确定性 | 22/24 | [0.742, 0.977] | 0.193 |

- 24次重复不足以证明名义95%覆盖已校准；这里是条件合成压力测试，不是人体准确度，也不是完整自由皮肤四维MCMC的重复覆盖实验。
- 另运行窄/中/宽材料先验敏感性比较，以及增加0.25 mm薄层的结构失配压力测试。结构失配的标称白化残差SSE≈250（42维），提示当前三组织模型不适用；不假定MCMC会覆盖未建模结构误差。
- 最初5阶材料求积漏掉窄后验，结果被拒绝；初版重要性积分的宽尾定量误差过大，也被替代。完整旧日志保留，selected_evidence=false。

## 验收状态与未取得的数据

- 27项测试通过，完整记录见 results_step6_recursive/validation_tests_final.txt。A的数据接口部分、B、C的合成部分、D和E的上述条件实验已运行。
- A的原始实测复算及C的实测噪声估计仍待MAT；完整四维MCMC重复覆盖率、真实校准迁移、硬件IQ/采样配置独立核验未完成。先验工程宽度及QC阈值尚未用独立材料/人体资料验证。
- 真实map/sample均返回calibration_missing，没有生成真实厚度候选或后验。真实g仍为未核实占位值，入口另设独立几何与重复稳定性校验。
- 后续需原始人体/空场MAT（每包小于32 MiB）、独立已知参考完整CIR及加载仿体验证、参考面/快时间原点/固件信息。准确度评价还需同部位厚度真值及subject/site/session映射；缺失保持空值。
- 厚度量分别为d_skin_mm、单侧d_fat_mm和派生d_cal_mm=2*(ds+df)。没有从姿态目录推断受试者、部位或标签。

## 证据目录与复现

所有路径相对项目根。源代码和CLI运行方法见 step6_recursive_bayes/README.md；固定版本见 requirements_step6_recursive.txt；各运行有manifest、config_resolved、hash/种子/BLAS、诊断和图。report_regeneration_manifest记录仅重算诊断/图表的版本，链数组未改动。

- audit: results_step6_recursive/20261025T033007Z_audit_a45099/
- physics: results_step6_recursive/20261025T031531Z_validate-physics_47961f/
- noise: results_step6_recursive/20261025T031647Z_estimate-noise_900d8a/
- known_targets: results_step6_recursive/20261025T032556Z_validate-statistics_eab0d8/
- recovery: results_step6_recursive/20261025T033019Z_validate-recovery_ab3c1b/
- fixed: results_step6_recursive/20261025T032343Z_recover-synthetic_1b8cff/
- uncertain: results_step6_recursive/20261025T032343Z_recover-synthetic_2ddef1/
- paired_fixed: results_step6_recursive/20261025T033056Z_recover-synthetic_ea5418/
- coverage: results_step6_recursive/20261025T032445Z_validate-coverage_6752ab/
- real_map: results_step6_recursive/20261025T031648Z_map_8c66f2/
- real_sample: results_step6_recursive/20261025T033007Z_sample_56cf37/
- smoke: results_step6_recursive/20261025T033203Z_recover-synthetic_e62e4c/

失败和中断运行由 results_step6_recursive/run_registry.json 标记，不混入验收统计。

交付仅含新增模块、配置、测试和结果，不包含或覆盖原始MAT、Step5 observations、旧TMM或图片。需在原项目根解压新增内容；材料ASCII路径可按本地实际位置修改。
