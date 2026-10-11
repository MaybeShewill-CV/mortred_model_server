# 第3步签字

- 问题: D-10
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 看 utilization 列为 N/A 的这一次成功探测的人，在 Linux 上、仓库根目录分开执行四条命令。验收 1 退出码 0，标准错误为空，标准输出原文五行依次是 `available=false`、`sample_count=1`、`util=-1`、`mortred-supervisor: gpu sampler offline (utilization N/A)`、`  GPU         ● FakeGPU  util n/a  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W`。验收 2 把假 `nvidia-smi` 的利用率列改成 `N/A`，退出码 0，标准错误为空，标准输出原文与验收 1 相同。验收 3 退出码 0，标准错误为空，标准输出原文五行依次是 `available=true`、`sample_count=1`、`util=37`、`mortred-supervisor: gpu sampler online (FakeGPU)`、`  GPU         ● FakeGPU  util 37%  vram 2.0G/8.0G (25%)  temp -1°C  pwr -1W`。验收 4 退出码 0，标准错误为空，标准输出原文四行依次是 `available=false`、`sample_count=0`、`mortred-supervisor: gpu sampler offline (no nvidia-smi / no GPU)`、`gpu_line=absent`。四条各自一次命令的退出码和输出原文就能判定冻结句：这次探测的 `available` 为 false，日志写出 offline，CLI 的利用率格是 `util n/a`。
