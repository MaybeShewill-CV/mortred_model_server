# Design P1-2 v1 (user-locked)
## Goal
Unify YOLO unmap↔NMS order: unmap letterbox then finalize/NMS (v5/v6/v7 order).
## Fix
Change yolov8 from NMS-then-unmap to unmap-then-NMS.
## Residual
Broader matting/seg/libface family audit parked — not in this change.
