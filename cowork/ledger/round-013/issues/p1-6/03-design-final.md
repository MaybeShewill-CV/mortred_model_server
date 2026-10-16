# Design P1-6 v1 (user-locked)
## Goal
Reject oversize encoded images from JPEG SOF / PNG IHDR before full imdecode.
## Fix
Add encoded_image_within_limits (JPEG via existing jpeg_full_dimensions + new PNG IHDR probe). Call before imdecode in image_input and base64 load paths.
