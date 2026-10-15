# 技术方案

- 问题: P2-23
- 版本: v1
- 对应列表: v1

## 边界

本方案只改英文与中文两份部署文档中 §5.2 验收预期与 §14.1 排障句，使其与 §10.7（自建 cpu pack；demo 为 TRT 种子；不发布默认 cpu 产品）一致。一次打开 `docs/deployment.md` 与 `docs/deployment.zh-cn.md`。

现行为：§5.2 写 cpu catalog「两个 `*_cpu` 条目 - mobilenetv2 / resnet」；§14.1 写「cpu 需要 `*_cpu` 配置」。中文对应句同样。仓库已无产品域 `*_cpu.toml` / `*_cpu_config.toml`；默认 `conf/packs/demo.toml` 为 TRT 种子。

锁定的可观察变化（队列 v1）：只读这两份部署文档：§5.2 / §14.1（及中文）不再要求随包 `*_cpu` 产品；与 §10.7「自建 cpu pack」一致。最小措辞改动，不整篇重写。

验收只打开文件。不拉起模型进程，不跑 GPU、calibrate 或压测。

## 方案

### 推荐

做法一：只改 §5.2 Expected/预期 与 §14.1 empty-catalog 行的修复措辞，指向 §10.7。

**英文 `docs/deployment.md`**

§5.2 由「cpu: the two `*_cpu` entries - mobilenetv2 / resnet」改为：catalog 按当前 profile 与 pack 列出；默认 demo pack 在 cpu 下为 supervisor+gateway only；无随包 `*_cpu` 产品；自建见 §10.7（demo 为 TRT seed）。

§14.1 empty catalog 修复列由「cpu needs the `*_cpu` configs」改为：核对 env 与 pack；cpu 需自建 pack（§10.7），不要求随包 `*_cpu` 产品。

**中文 `docs/deployment.zh-cn.md`** 对应 §5.2 / §14.1 同样最小改写。

§10.7 正文、§8.2、其它章节、compose 命令表保持原样。

代价：对照过旧「两个 `*_cpu`」句的笔记要对齐 §10.7。没有进程代价。

### 备选

做法二：删除 §5.2 整句 cpu 预期、只留 health OK。读者仍不知空 catalog 是否正常。否决。

做法三：恢复随包 `*_cpu` 产品配置以迁就旧文档。与 `ea22867d` 与 §10.7 相反。否决。

做法四：整篇重写 deployment。队列要求最小措辞。否决。

## 改动面

只改 docs 模块两份部署文档的 §5.2 与 §14.1 相关句。

- `docs/deployment.md`
- `docs/deployment.zh-cn.md`

## 不改清单

§10.7 正文；§8.2；教程（P2-22）；how-to；api-contract；openapi；README；产品/server 代码；`conf/packs/demo.toml`；gateway / supervisor。

## 契约

HTTP、OpenAPI、tensor、产品 toml 不变。仅文档预期与排障措辞。

## 性能与资源

不拉起进程，不生成 engine，不改 worker_nums，不跑压测。

## 验收标准

1. 打开 `docs/deployment.md` §5.2：不再要求随包两个 `*_cpu` 产品条目；表述与 §10.7 一致（自建 cpu pack；demo 为 TRT seed）。
2. 同文件 §14.1：empty catalog 修复不再写「cpu needs the `*_cpu` configs」。
3. 打开 `docs/deployment.zh-cn.md` §5.2 / §14.1：同样不再要求随包 `*_cpu` 产品；与 §10.7 一致。
4. diff 只出现在这两份 deployment。不触及教程、how-to、代码。

## 黑盒步骤

1. QA 在 worktree 执行自测，退出码 0。
2. QA 用 diff 确认只改两份 deployment。
3. 按验收标准打开文件核对。

## 自测命令

在 worktree 根用 python3 断言：两份文件均不再出现要求随包 `*_cpu` 产品条目的旧 §5.2 / §14.1 措辞；均仍含 §10.7 标题或「no default cpu product」/「不提供默认 cpu 产品」；§5.2 仍含 health 预期。打印 `p2-23-deployment-cpu-expect ok`。

## RD 实现时注意

- 只改 §5.2 与 §14.1 相关句；不得整篇重写。
- 不得只改英或只改中。
- 不改 §10.7 正文、教程、产品代码。
- 自测 root 换成 worktree。不升 P0/P1，不扩 scope。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| PM | 同意 | 无 | v1 | cowork/ledger/round-003/issues/p2-23/03-discussion/v1-sign-pm.md |
| RD | 同意 | 无 | v1 | cowork/ledger/round-003/issues/p2-23/03-discussion/v1-sign-rd.md |
| 技术负责人 | 同意 | 无 | v1 | cowork/ledger/round-003/issues/p2-23/03-discussion/v1-sign-tech-lead.md |
