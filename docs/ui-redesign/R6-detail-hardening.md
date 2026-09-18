# R6 — 用户反馈缺陷修复：文件夹上传 / 时间显示 / 细节加固

**日期**：2026-09-18　**前置**：[R5](R5-hero-and-final.md)
**来源**：用户实测反馈的两个具体 bug + 由此展开的细节打磨。

## 迭代前（R5 定稿）

| 视图 | 截图 |
|---|---|
| Workbench（bench 上传区） | ![before](shots/r8-final-workbench-day.png) |

## 迭代后

| 视图 | 截图 |
|---|---|
| Workbench（文件夹 15 图入队：chips 封顶 + 计数 toast） | ![after](shots/r9-fix-workbench-folderupload-day.png) |

## Bug 1：选 folder 只上传一张图

**根因**：`addFiles()` 只认 `f.type.startsWith("image/")`。经
`webkitdirectory` 选文件夹时，很多系统（尤其 Linux/WSL——本项目的主力
部署环境）返回的 `File.type` 是**空字符串**，于是整folder的图片被静默
过滤，只剩个别碰巧带 MIME 的文件。

**修复**：
- 新增 `isImageFile()`：MIME 为 `image/*` **或** 扩展名命中
  （png/jpg/jpeg/bmp/gif/webp/ppm/pgm/pbm/tif/tiff/avif）即接受；
  `.DS_Store`、`notes.txt` 等依旧正确过滤。
- `<input webkitdirectory>` 补 `directory` 属性（跨浏览器目录选择）。
- 反馈细节：入队成功 toast「Added N images to the bench」；空选择提示
  「No images found in the selection」。
- 布局防护：文件 chips 封顶 12 个 + 「+ N more queued」计数胶囊
  （整个队列保留，全部参与发送——百图文件夹不再撑爆试验台）。

**验证**（浏览器注入 15 个空 MIME `.jpg` + `.DS_Store` + `.txt`）：
入队 15 ✓、垃圾过滤 ✓、chips 12 + 「+ 3 more queued」✓、toast ✓。

## Bug 2：GPU telemetry 显示「4.8333333333分钟」

**根因**：`gpu-meta` 拼接 `(winS / 60) + " min"`，未取整。样本数 >120
（真实服务器的长窗口）时浮点原样泄漏。

**修复**：新增 `fmtWindow()`——<2 min 显示秒；分钟值 <10 保留 1 位小数、
≥10 取整（`5 min`、`2.5 min`、`25 min`）。同源加固：
- 图表十字线 tooltip 的 `fmt()` 统一 `Math.round`（温度/功率不再可能
  出现 `64.33333°C`）；
- 工作台 GPU 温度读数 `Math.round`。

**验证**：mock 窗口扩至 150 样本（5 min 路径）→
`1 gpu · last 5 min · 2 s poll · hover to inspect` ✓。

## 顺带发现并修复的第三个 bug：深链误报

验证 Bug 1 时暴露：直接打开 `/#/model/yolov8` 时，启动头 2 秒 catalog
未到，`renderCurrentView` 误报 toast「Model not found: yolov8」随后又
跳进工作台。修复：新增 `state.catalogReady` 门闩，只有首轮
catalog+status 成功后才允许报 not-found。

## 回归烟测（全部通过）

palette 10 行 ✓；主题 day→night→day ✓；过滤 running→6 卡 / all→12 卡 ✓；
版本徽章 ✓；深链 0 误报 toast ✓。

## 评分影响

本轮为缺陷修复型迭代，不改变视觉体系（视觉五项维持 9.1–9.4），
直接提升 ⑦ 一致性与状态设计（消除了 3 个真实用户可感知的瑕疵：
静默丢图、浮点时间、启动误报）与 ⑥ 信任感（文件夹批量场景现在有
明确的数量反馈）。功能性细节的完备是「产品级」叙事的必要条件。
