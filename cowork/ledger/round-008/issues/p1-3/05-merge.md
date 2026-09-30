# 合入

- 问题: P1-3
- 分支: cowork/r08-p1-3-yolov8-inf
- 工作树: /tmp/mortred-cowork-r08-p1-3
- 问题分支提交: 7044420c
- QA 结论: pass
- 重跑: `scripts/run_tests.sh build/full -R 'object_detection_output_contract'` 退出码 0（1/1 Passed，6 tests）
- 偏离: 几何宽高断言 10 改为 5，已签字同意
- merge commit: d80c104b6b62b610c7ba5b650410428a93124ec2
