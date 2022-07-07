# About Model Configuration

All model's configurations are stored in $PROJECT_ROOT_DIR/conf/model folder.

## Common Configuration

Use mobilenetv2's model configuration for example
![common_model_config](../resources/images/common_model_config_example.png)

**model_file_path:** model's weights file path

**model_threads_num:** computing threads nums when cpu backend was used. recommand to use the amount of cpu core

**compute_backend:** compute backend only "cpu" and "cuda" was supported for now

## Special Model Configuration

### For Image Objection Model Configuration

**model_score_threshold:** obj-bbox's score threshold only keep objs with score higher than this

**model_nms_threshold:** nms threshold used for merging bboxes aimed at the same objs

**model_keep_top_k:** maxinum count of kept objects

**model_class_nums:** total number of categories the model can recognize


### SuperPoint Model Configuration

**model_nms_threshold:** only one feature point can be kept in radius of this threshold

**max_track_length:** only track images which are closer than this distance

**nn_dis_thresh:** thoses points whose feature distances are smaller than this will be regarded as the same point

