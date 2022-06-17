/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: enlightengan_fast_batch_process.cpp
* Date: 22-6-17
************************************************/

// enlightengan fast batch process tool

#include <random>

#include <glog/logging.h>
#include <toml/toml.hpp>
#include <opencv2/opencv.hpp>

#include "common/cv_utils.h"
#include "common/file_path_util.h"
#include "common/time_stamp.h"
#include "models/model_io_define.h"
#include "factory/enhancement_task.h"

using morted::common::FilePathUtil;
using morted::common::Timestamp;
using morted::common::CvUtils;
using morted::models::io_define::common_io::mat_input;
using morted::models::io_define::enhancement::std_enhancement_output;
using morted::factory::enhancement::create_enlightengan_enhancementor;


