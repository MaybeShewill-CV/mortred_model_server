# 关闭记录

- 轮次: 4
- 已关闭: P0-1、P0-2、P0-3、P0-4、P1-1、P1-2、P1-3、P1-4、P1-5
- 阻塞: 无
- 未阻塞的开放项: 相对本轮起点减少

## 已关闭的问题陈述
| id | 模块 | 问题陈述 |
| --- | --- | --- |
| P0-1 | conf/ops | TLS 私钥在 conf/local/edge/tls；停跟踪+gitignore+init-edge 注释+consistency |
| P0-2 | conf/control | Gateway rate_limit 默认开启；公网 listen 全关时 fail-closed |
| P0-3 | control | occupancy 关闭须 MORTRED_OCCUPANCY_ALLOW_DISABLE |
| P0-4 | models | GPU zero-copy 保留 input.params |
| P1-1 | server/control | FixedWindow vs GCRA 代码内文档 |
| P1-2 | control | nvidia-smi reaper 竞态硬化 |
| P1-3 | server | openapi_doc.h 瘦身为运行时加载 |
| P1-4 | control | ApiKey max_inflight 首切片 |
| P1-5 | build | CMake VERSION 1.0.0 |

## 仍开放的问题
无

## 附录
| id | 陈述 |
| --- | --- |
| 用户删除-原P1-1 | 产品 max_batch_size=1 |
| 用户删除-原P1-5 | GPU CI 非可移植 |
| stash | r4-wip-param-occupancy-async-before-designs 仍保留 async/batch 等未入本轮项，勿丢 |

## 新入列
无

## Merge SHAs
| id | issue commit | merge commit |
| --- | --- | --- |
| P0-1 | e4e13b61 | e1e2cb1a |
| P0-2 | 57cc6d74 | 03cbde91 |
| P0-3 | f6e5dbf4 | a88866c3 |
| P0-4 | 75084dc5 | c4e74ea4 |
| P1-1 | c56cdf9d | a8504f13 |
| P1-2 | 18fbe870 | 00073149 |
| P1-3 | c713915c | e8c02df6 |
| P1-4 | 3e2c1a0a | ee3b1cf3 |
| P1-5 | 53c4988d | 2ac5ae2c |