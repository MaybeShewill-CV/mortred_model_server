/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: model_io_define.h
 * Date: 22-6-7
 ************************************************/

#ifndef MM_AI_SERVER_MODEL_IO_DEFINE_H
#define MM_AI_SERVER_MODEL_IO_DEFINE_H

// Compatibility aggregate. Include only the task header you need from
// src/models/io/ in new code; this file keeps the existing call sites
// compiling unchanged. It must stay a pure include list - no types.

#include "models/io/classification.h"
#include "models/io/clip.h"
#include "models/io/common_input.h"
#include "models/io/diffusion.h"
#include "models/io/enhancement.h"
#include "models/io/feature_point.h"
#include "models/io/matting.h"
#include "models/io/mono_depth_estimation.h"
#include "models/io/object_detection.h"
#include "models/io/ocr.h"
#include "models/io/scene_segmentation.h"
#include "models/io/segment_anything.h"

#endif // MM_AI_SERVER_MODEL_IO_DEFINE_H
