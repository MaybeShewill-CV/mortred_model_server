# Supervisor 控制台 — 视觉迭代记录（Screenshot Gallery）

> 本文档记录 `fix/ui-phosphor-redesign` 分支上 Supervisor UI 的每一轮美学深化迭代：
> 改动内容、八维度评分（运维控制台权重）与当轮渲染截图。
> 截图由本地 mock（12 模型 / GPU 采样 / 日志流）驱动真实浏览器渲染所得。
>
> 评分维度与权重：人格与记忆点 10% · 概念清晰度 10% · 色彩系统 15% · 字体与排版 10%
> · 表面与材质 15% · 密度与信息层级 15% · 状态可见性 15% · 动效与细节工艺 10%
>
> **退出条件**：所有维度 ≥ 8.5，人格与记忆点 ≥ 9.0，加权总分 ≥ 9.0。

---

## Round 1 — 色彩纪律 / 材质分层 / Mission-Control 布局（2026-09-17）

### 改动清单

| 维度目标 | 改动 |
|---|---|
| 色彩系统 | 分类色板重制为统一感知亮度的色相家族（teal/cyan/blue/violet/amber/rose…，同明度同饱和带）；语义色（acc/warn/err）与身份色彻底分离；状态 chips 携带各自语义色 |
| 字体排版 | 全站字号阶梯 token（10/10.5/11.5/13/15/18）；数字场景全面 `tabular-nums` 列对齐 |
| 表面材质 | 日志面板与拖放区改内嵌井（inset shadow）；卡片内顶部高光 + 悬停投影；SVG 噪点材质（data-URI，零资源依赖）+ 页面暗角 |
| 密度层级 | Overview 固定三段式布局：GPU / 舰队（弹性滚动）/ 遥测带一屏尽收；新增状态汇总过滤 chips（running/starting/failed/stopped + 计数，可点选过滤）与分类 chips 同行 |
| 状态可见性 | 标签页标题实时 live 计数；舰队卡片键盘可达（tabindex + Enter/Space + aria-label）；GPU 图表十字线取样器（hover 显示 util/vram/pwr/temp）；过滤无结果的空态文案 |
| 动效工艺 | 磷光角标（targeting brackets）回归 hover；视图切换过渡动画；命令面板弹入动画 + 命令类型图标（▸/⟳/◈）+ 空态；`prompt()` 全面替换为样式化 token 对话框（毛玻璃背景、聚焦环、Enter/Esc 语义）；全局 kbd 芯片组件 + 顶栏跨平台快捷键提示（⌘K/Ctrl K） |

### 当轮截图

| 视图 | 截图 |
|---|---|
| Overview 1600（固定三段式 + 双过滤行） | ![round1 overview](images/ui/round1-overview-1600.png) |
| GPU 十字线取样器 | ![round1 crosshair](images/ui/round1-gpu-crosshair.png) |
| Workbench（内嵌井日志面板） | ![round1 workbench](images/ui/round1-workbench.png) |
| 命令面板（图标 + kbd 页脚） | ![round1 palette](images/ui/round1-palette.png) |
| Token 对话框 | ![round1 token](images/ui/round1-token-clean.png) |

### 评分（Round 1）

| 维度 | R0 基线 | R1 | 依据 |
|---|---|---|---|
| 人格与记忆点 | 9.00 | **9.00** | 磷光签名保留，角标/十字线/噪点强化身份，但尚无新增"标志性瞬间" |
| 概念清晰度 | 8.50 | **9.00** | 固定三段式兑现 "Mission Control" 叙事，一屏尽收 |
| 色彩系统 | 7.50 | **8.50** | 色相家族统一感知亮度；但图表/JS 内仍有少量裸 hex 未与 token 同源 |
| 字体与排版 | 7.50 | **8.50** | 阶梯 token + tnum 全面铺开；Workbench 标题层级仍可再进一档 |
| 表面与材质 | 7.50 | **8.50** | 井/投影/噪点/暗角成体系；live 卡描边仍是纯色而非渐变 |
| 密度与信息层级 | 8.00 | **8.75** | 双过滤行 + 固定布局；GPU 面板头部信息密度尚可再压 |
| 状态可见性 | 8.00 | **8.75** | title 计数/键盘可达/十字线；断链态仍是简单降透明度 |
| 动效与细节工艺 | 8.00 | **8.75** | 弹入/过渡/空态/kbd 体系；数字无 tween，toast 离场生硬 |
| **加权总分** | 8.00 | **8.70** | 未达退出条件（≥9.0）→ **继续 Round 2** |

### Round 2 计划

色彩 token 同源化（JS↔CSS）、live 卡渐变描边、日志面板 CRT 扫描线、boot 序列、
断链覆盖层、hud 数字 tween、toast 离场动画、palette 行 stagger、全局 focus 统一。
