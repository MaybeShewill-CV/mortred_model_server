# 合入

- 问题: P1-1
- 分支: cowork/r10-p1-1-clip-boxes
- worktree: /mnt/g/Codex/mortred_wt_r10b
- 问题分支提交: 90d3392e60af063560709f8c32d18a048e576736
- QA 结论: pass
- 重跑: 编排方在 worktree 重跑 QA 三条命令，退出码与报告一致（8 / 15 / 13 全过，exit 0）；`clip_boxes` grep 与 QA 报告逐字一致；worktree 内两文件与提交 blob 的 sha1 已由 QA 比对相同
- 命令: `git merge --no-ff cowork/r10-p1-1-clip-boxes -m "Merge branch cowork/r10-p1-1-clip-boxes (Round 10 P1-1)"`
- merge commit: db231f5c8cbad80ddba0d96e565d80cb82b7a44b

## 编排方重跑记录

| 命令 | 退出码 | 输出 |
| --- | --- | --- |
| `./_bin/detection_params_unittest` | 0 | `[  PASSED  ] 8 tests.` |
| `./_bin/detector_common_unittest` | 0 | `[  PASSED  ] 15 tests.` |
| `./_bin/object_detection_output_contract_unittest` | 0 | `[  PASSED  ] 13 tests.` |
| `grep -rn clip_boxes src/ test/ conf/ docs/openapi.json` | 0 | 4 命中：`detection_params.h:49,51` + `test/detection_params_unittest.cc:96,101` |
| `grep -rn clip_boxes conf/model/object_detection/` | 1 | 0 命中（无出货 TOML 使用该键） |

合入前 dry-run（`git merge-tree --write-tree`）输出 tree `e5110b43`，无冲突；合入后基线工作树干净。

## 合入内容

| 文件 | 变化 |
| --- | --- |
| `src/models/object_detection/detection_params.h` | 删除无效的 `clip_boxes` 字段与其旧解析分支；按本文件既有"已移除键"约定（`input_node_size`）新增显式拒绝，错误文案说明"detections are always clipped to the source image after letterbox unmap" |
| `test/detection_params_unittest.cc` | 新增 `RejectsClipBoxesKey`（true/false/`"yes"` 三态均拒）与 `ParsesMinBoxAreaPx`（默认 5.0、12.5、0 的邻位守卫） |

合计 35 insertions / 11 deletions，产品语义零改动（`min_box_area_px`、`passes_min_box_area`、各检测器过滤位置、`unmap_letterbox_bbox` 恒定夹取全部未动）。

## 清理

合入记录写完后删除该 worktree。注意：该 worktree 的 git 注册已在此前一次 `git worktree prune` 中被清除（见 `discovered.md`），因此不走 `git worktree remove`，直接确认无目录联接后删除目录。
