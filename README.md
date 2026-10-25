# UWB Step6：递归反射与贝叶斯厚度反演

本仓库保存新增 Step6 模块、配置、测试、实施报告及验证交付包。原始雷达 MAT、Step5 observations、两篇论文和历史项目没有上传。

## 下载交付

- [代码 ZIP](deliverables/step6_recursive_bayes_code.zip)
- [验证证据 ZIP](deliverables/step6_recursive_bayes_evidence.zip)
- [中文实施报告](STEP6_IMPLEMENTATION_REPORT_zh.md)
- [模块说明与运行命令](step6_recursive_bayes/README.md)

打开 ZIP 页面后点击 **Download raw file** 下载，也可从本页 Code → Download ZIP 下载整个仓库。

## 当前验证结果

27 项测试通过。固定材料及材料不确定性 MCMC 均通过收敛诊断；固定材料后验与密集积分通过比较。每组 24 次条件合成覆盖率实验和失败/修正日志见实施报告与证据包。所有报告中的厚度数值均为合成实验。

真实厚度反演因缺独立实测校准返回 `calibration_missing`。原始 MAT 复算、实测空场协方差和人体准确度尚未验收。证据包中的 manifest/audit 含处理元信息和输入文件 hash，不含原始人体 CIR 或真实 observations 数组；后验样本和预测图均来自合成数据。

## 使用

这些是已有项目的新增模块。请与原 Step5 observations、构建脚本、IT’IS ASCII 和历史 TMM 放在同一项目根下，并按实际位置修改 JSON 配置中的材料路径。

```bash
python -m pip install -r requirements_step6_recursive.txt
python -m step6_recursive_bayes.cli audit --config configs/step6_audit.json
python -m step6_recursive_bayes.cli recover-synthetic --config configs/step6_synthetic_uncertain.json
python -m pytest tests_step6_recursive -q
```

不要将空场或历史人体 Psys 拟合当成独立校准，不要将合成恢复精度解释为人体厚度准确度。
