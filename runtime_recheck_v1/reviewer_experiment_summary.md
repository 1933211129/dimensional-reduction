# Reviewer 实验补充结果总结

## 0. 实验口径
- 外层交叉验证折数：`2`。
- 为在统一计算预算下完成可复现比较，本次统一使用数据采样：`sample_fraction=1.0`，`max_objects=40`。

## 1. Ranking Loss 定义与实现核验
- 手工构造样例与代码实现的差值是否为 0：`True`。
- 核验口径：相关标签 vs 不相关标签，分母为 `|Y_t| * |Y_bar_t|`。
- `single-sample-perfect-order`：manual=0.000000，implementation=0.000000。
- `single-sample-complete-error`：manual=1.000000，implementation=1.000000。
- `two-sample-mixed`：manual=0.583333，implementation=0.583333。

## 2. Baseline 缺失值公平性
- 支持原生缺失值处理的方法保持 `native` 路径；需要插补的 baseline 在训练折内对 `KNN/EM/MissForest` 做独立调优，主结果只保留每个任务点的最佳配置。
- `MCB-AR` 最终最佳插补分布：native=9，KNN=0，EM=0，MissForest=0，mixed=0。
- `MFS-MCDM` 最终最佳插补分布：native=0，KNN=2，EM=2，MissForest=2，mixed=3。
- `FIMF` 最终最佳插补分布：native=0，KNN=0，EM=1，MissForest=0，mixed=8。
- `ML-CSFS` 最终最佳插补分布：native=0，KNN=3，EM=2，MissForest=0，mixed=4。

## 3. 主结果与显著性
- 以下计数基于 9 个任务点（3 个数据集 × 3 个缺失率）。
- 对 `MCB-AR`：F1 更优/更差/持平 = 3/4/2；Ranking Loss 更优/更差/持平 = 6/2/1。
- 对 `MFS-MCDM`：F1 更优/更差/持平 = 5/4/0；Ranking Loss 更优/更差/持平 = 5/4/0。
- 对 `FIMF`：F1 更优/更差/持平 = 8/1/0；Ranking Loss 更优/更差/持平 = 4/5/0。
- 对 `ML-CSFS`：F1 更优/更差/持平 = 8/1/0；Ranking Loss 更优/更差/持平 = 2/7/0。
- Wilcoxon signed-rank test 结果：
  - `f1_score` vs `MCB-AR`：p=0.90625，Ours=0.290292，Baseline=0.290718。
  - `f1_score` vs `MFS-MCDM`：p=0.511719，Ours=0.290292，Baseline=0.268101。
  - `f1_score` vs `FIMF`：p=0.0390625，Ours=0.290292，Baseline=0.223709。
  - `f1_score` vs `ML-CSFS`：p=0.0507812，Ours=0.290292，Baseline=0.219983。
  - `ranking_loss` vs `MCB-AR`：p=0.546875，Ours=0.443409，Baseline=0.445738。
  - `ranking_loss` vs `MFS-MCDM`：p=0.496094，Ours=0.443409，Baseline=0.453195。
  - `ranking_loss` vs `FIMF`：p=0.734375，Ours=0.443409，Baseline=0.433598。
  - `ranking_loss` vs `ML-CSFS`：p=0.0390625，Ours=0.443409，Baseline=0.423376。
  - `coverage` vs `MCB-AR`：p=0.640625，Ours=6.717871，Baseline=6.744600。
  - `coverage` vs `MFS-MCDM`：p=0.25，Ours=6.717871，Baseline=6.916259。
  - `coverage` vs `FIMF`：p=0.652344，Ours=6.717871，Baseline=6.835392。
  - `coverage` vs `ML-CSFS`：p=0.300781，Ours=6.717871，Baseline=6.498776。
  - `hamming_loss` vs `MCB-AR`：p=0.1875，Ours=0.222031，Baseline=0.212893。
  - `hamming_loss` vs `MFS-MCDM`：p=0.285156，Ours=0.222031，Baseline=0.207281。
  - `hamming_loss` vs `FIMF`：p=0.359375，Ours=0.222031，Baseline=0.209895。
  - `hamming_loss` vs `ML-CSFS`：p=0.734375，Ours=0.222031，Baseline=0.226638。
  - `reduction_time` vs `MCB-AR`：p=0.300781，Ours=0.022234，Baseline=0.027524。
  - `reduction_time` vs `MFS-MCDM`：p=0.00390625，Ours=0.022234，Baseline=0.002931。
  - `reduction_time` vs `FIMF`：p=0.00390625，Ours=0.022234，Baseline=0.429606。
  - `reduction_time` vs `ML-CSFS`：p=0.0078125，Ours=0.022234，Baseline=0.834211。

## 4. 运行时间与大规模适用性
- 主结果中的本地数据运行时间现在已单独导出到 `runtime_local.csv`。
- 审稿回复中建议把 `avg_reduction_time` 作为主效率指标，因为它表示在最终选定配置下的纯约简/选择时间。
- `avg_selection_pipeline_time` 则表示训练折内调参与最终重拟合选择器的总开销，可作为补充说明而不应与纯约简时间混为一谈。
- `Ours` 在 9 个任务点上的平均纯约简时间：0.022234 s。
- `MCB-AR` 在 9 个任务点上的平均纯约简时间：0.027524 s。
- `MFS-MCDM` 在 9 个任务点上的平均纯约简时间：0.002931 s。
- `FIMF` 在 9 个任务点上的平均纯约简时间：0.429606 s。
- `ML-CSFS` 在 9 个任务点上的平均纯约简时间：0.834211 s。
- `Ours` 在 9 个任务点上的平均选择总开销：0.022234 s。
- `MCB-AR` 在 9 个任务点上的平均选择总开销：0.027524 s。
- `MFS-MCDM` 在 9 个任务点上的平均选择总开销：24.121350 s。
- `FIMF` 在 9 个任务点上的平均选择总开销：27.975320 s。
- `ML-CSFS` 在 9 个任务点上的平均选择总开销：36.841549 s。

## 5. 面向审稿意见的可直接回复点
- 对审稿人 2 关于 Ranking Loss 的质疑：代码实现已按标准 relevant-vs-irrelevant 定义核验，当前实现与手工计算一致。
- 对审稿人 1 关于 baseline fairness 的质疑：所有 baseline 现在都在训练折内对 KNN、EM、MissForest 做公平调优，主表不再固定单一均值插补。
- 对审稿人 1 关于 significance 的要求：已补充基于 9 个任务点的 Wilcoxon signed-rank test。
- 对审稿人 1 关于 runtime / large-scale applicability 的质疑：已补本地运行时间表和 synthetic scaling 结果，分别回答实际耗时与规模增长趋势。
- 对审稿人 1 关于 case study generality 的质疑：除案例分析外，当前结果已覆盖 birds、scene、yeast 三个公开多标签数据集与 3 个缺失率设置。
