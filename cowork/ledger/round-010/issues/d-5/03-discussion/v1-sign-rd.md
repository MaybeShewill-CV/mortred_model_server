# 第3步签字

- 问题: D-5 deployment.md §15 FAQ 结构损坏
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `docs/deployment.md` 里 `## 15. FAQ` 到 `## 16. Acceptance Gates` 之间，前六条 `<details>` 已经各自有 `<summary>` 并且闭合。`Where do I set worker_nums?` 的 `</details>` 之后、节末 `---` 之前，现文就是无 summary 的三段英文，并以一个多余的 `</details>` 结束。这一节现在是 6 个 `<details>`、7 个 `</details>`。按正文删掉这段残留和那个多余结束标签之后，标签序列是 `<details>`、`<summary>`、`</summary>`、`</details>` 重复 6 次；去掉这 6 块和标题 `## 15. FAQ` 后，非空行只剩单独成行的 `---`。六条 summary 与六段正文，和契约原文、自测脚本、现文一致。`Fetch on any machine with python` 与 `python-less` 只出现在这段残留里，删掉后全文不再含这两句。`### 9.4 Offline environments` 到 `## 10. Machine pack and TensorRT` 之前已经含有 `Fetch `weights/` on a networked machine → copy to the target → run` 和 `` `fetch_weights.py --check`. Dependencies work the same way (`--offline DIR`). ``。`### 14.3 Deep dives` 到 `## 15. FAQ` 之前已经是同一标签循环重复 2 次，两条 summary 与验收第 7 条一致。验收第 1 条到第 7 条都已编号，只读该文件就能判定。自测是仓库根目录、能运行 `python3` 的 shell 里的 heredoc，不启动 gateway 或 supervisor；按该删除执行后退出码为 0。本条只改文档结构，推理热路径仍走现有预处理与拷贝。
