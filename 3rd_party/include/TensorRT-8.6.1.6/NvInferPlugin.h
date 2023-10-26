/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NV_INFER_PLUGIN_H
#define NV_INFER_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPluginUtils.h"
//!
//! \file NvInferPlugin.h
//!
//! This is the API for the Nvidia provided TensorRT plugins.
//!

extern "C"
{
    //!
    //! \brief Create a plugin layer that fuses the RPN and ROI pooling using user-defined parameters.
    //! Registered plugin type "RPROI_TRT". Registered plugin version "1".
    //! \param featureStride Feature stride.
    //! \param preNmsTop Number of proposals to keep before applying NMS.
    //! \param nmsMaxOut Number of remaining proposals after applying NMS.
    //! \param iouThreshold IoU threshold.
    //! \param minBoxSize Minimum allowed bounding box size before scaling.
    //! \param spatialScale Spatial scale between the input image and the last feature map.
    //! \param pooling Spatial dimensions of pooled ROIs.
    //! \param anchorRatios Aspect ratios for generating anchor windows.
    //! \param anchorScales Scales for generating anchor windows.
    //!
    //! \return Returns a FasterRCNN fused RPN+ROI pooling plugin. Returns nullptr on invalid inputs.
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use RPROIPluginCreator::createPlugin() to create an instance of
    //! "RPROI_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createRPNROIPlugin(int32_t featureStride, int32_t preNmsTop, int32_t nmsMaxOut,
        float iouThreshold, float minBoxSize, float spatialScale, nvinfer1::DimsHW pooling,
        nvinfer1::Weights anchorRatios, nvinfer1::Weights anchorScales);

    //!
    //! \brief The Normalize plugin layer normalizes the input to have L2 norm of 1 with scale learnable.
    //! Registered plugin type "Normalize_TRT". Registered plugin version "1".
    //! \param scales Scale weights that are applied to the output tensor.
    //! \param acrossSpatial Whether to compute the norm over adjacent channels (acrossSpatial is true) or nearby
    //! spatial locations (within channel in which case acrossSpatial is false).
    //! \param channelShared Whether the scale weight(s) is shared across channels.
    //! \param eps Epsilon for not dividing by zero.
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use NormalizePluginCreator::createPlugin() to create an instance of
    //! "Normalize_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createNormalizePlugin(
        nvinfer1::Weights const* scales, bool acrossSpatial, bool channelShared, float eps);

    //!
    //! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all
    //! dimensions (H x W). PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer.
    //! Registered plugin type "PriorBox_TRT". Registered plugin version "1".
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use PriorBoxPluginCreator::createPlugin() to create an instance of
    //! "PriorBox_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createPriorBoxPlugin(nvinfer1::plugin::PriorBoxParameters param);

    //!
    //! \brief The Grid Anchor Generator plugin layer generates the prior boxes of
    //! designated sizes and aspect ratios across all dimensions (H x W) for all feature maps.
    //! GridAnchorParameters defines a set of parameters for creating the GridAnchorGenerator plugin layer.
    //! Registered plugin type "GridAnchor_TRT". Registered plugin version "1".
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use GridAnchorPluginCreator::createPlugin() to create an instance of
    //! "GridAnchor_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createAnchorGeneratorPlugin(
        nvinfer1::plugin::GridAnchorParameters* param, int32_t numLayers);

    //!
    //! \brief The DetectionOutput plugin layer generates the detection output based on location and confidence
    //! predictions by doing non maximum suppression. DetectionOutputParameters defines a set of parameters for creating
    //! the DetectionOutput plugin layer. Registered plugin type "NMS_TRT". Registered plugin version "1".
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use NMSPluginCreator::createPlugin() to create an instance of "NMS_TRT"
    //! version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createNMSPlugin(nvinfer1::plugin::DetectionOutputParameters param);

    //!
    //! \brief The Reorg plugin reshapes input of shape CxHxW into a (C*stride*stride)x(H/stride)x(W/stride) shape, used
    //! in YOLOv2. It does that by taking 1 x stride x stride slices from tensor and flattening them into
    //! (stride x stride) x 1 x 1 shape. Registered plugin type "Reorg_TRT". Registered plugin version "1".
    //! \param stride Strides in H and W, it should divide both H and W. Also stride * stride should be less than or
    //! equal to C.
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use ReorgPluginCreator::createPlugin() to create an instance of
    //! "Reorg_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createReorgPlugin(int32_t stride);

    //!
    //! \brief The Region plugin layer performs region proposal calculation: generate 5 bounding boxes per cell (for
    //! yolo9000, generate 3 bounding boxes per cell). For each box, calculating its probablities of objects detections
    //! from 80 pre-defined classifications (yolo9000 has 9416 pre-defined classifications, and these 9416 items are
    //! organized as work-tree structure). RegionParameters defines a set of parameters for creating the Region plugin
    //! layer. Registered plugin type "Region_TRT". Registered plugin version "1".
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use RegionPluginCreator::createPlugin() to create an instance of
    //! "Region_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createRegionPlugin(nvinfer1::plugin::RegionParameters params);

    //!
    //! \brief The BatchedNMS Plugin performs non_max_suppression on the input boxes, per batch, across all classes.
    //! It greedily selects a subset of bounding boxes in descending order of
    //! score. Prunes away boxes that have a high intersection-over-union (IOU)
    //! overlap with previously selected boxes. Bounding boxes are supplied as [y1, x1, y2, x2],
    //! where (y1, x1) and (y2, x2) are the coordinates of any
    //! diagonal pair of box corners and the coordinates can be provided as normalized
    //! (i.e., lying in the interval [0, 1]) or absolute.
    //! The plugin expects two inputs.
    //! Input0 is expected to be 4-D float boxes tensor of shape [batch_size, num_boxes,
    //! q, 4], where q can be either 1 (if shareLocation is true) or num_classes.
    //! Input1 is expected to be a 3-D float scores tensor of shape [batch_size, num_boxes, num_classes]
    //! representing a single score corresponding to each box.
    //! The plugin returns four outputs.
    //! num_detections : A [batch_size] int32 tensor indicating the number of valid
    //! detections per batch item. Can be less than keepTopK. Only the top num_detections[i] entries in
    //! nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid.
    //! nmsed_boxes : A [batch_size, max_detections, 4] float32 tensor containing
    //! the co-ordinates of non-max suppressed boxes.
    //! nmsed_scores : A [batch_size, max_detections] float32 tensor containing the
    //! scores for the boxes.
    //! nmsed_classes :  A [batch_size, max_detections] float32 tensor containing the
    //! classes for the boxes.
    //!
    //! Registered plugin type "BatchedNMS_TRT". Registered plugin version "1".
    //!
    //! The batched NMS plugin can require a lot of workspace due to intermediate buffer usage. To get the
    //! estimated workspace size for the plugin for a batch size, use the API `plugin->getWorkspaceSize(batchSize)`.
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use BatchedNMSPluginCreator::createPlugin() to create an instance of
    //! "BatchedNMS_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createBatchedNMSPlugin(nvinfer1::plugin::NMSParameters param);

    //!
    //! \brief The Split Plugin performs a split operation on the input tensor. It
    //! splits the input tensor into several output tensors, each of a length corresponding to output_lengths.
    //! The split occurs along the axis specified by axis.
    //! \param axis The axis to split on.
    //! \param output_lengths The lengths of the output tensors.
    //! \param noutput The number of output tensors.
    //!
    //! \deprecated Deprecated in TensorRT 8.5 along with the "Split" plugin. Use INetworkDefinition::addSlice() to add
    //! slice layer(s) as necessary to accomplish the required effect.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createSplitPlugin(int32_t axis, int32_t* output_lengths, int32_t noutput);

    //!
    //! \brief The Instance Normalization Plugin computes the instance normalization of an input tensor.
    //! The instance normalization is calculated as found in the paper https://arxiv.org/abs/1607.08022.
    //! The calculation is y = scale * (x - mean) / sqrt(variance + epsilon) + bias where mean and variance
    //! are computed per instance per channel.
    //! \param epsilon The epsilon value to use to avoid division by zero.
    //! \param scale_weights The input 1-dimensional scale weights of size C to scale.
    //! \param bias_weights The input 1-dimensional bias weights of size C to offset.
    //!
    //! \deprecated Deprecated in TensorRT 8.5. Use InstanceNormalizationPluginCreator::createPlugin() to create an
    //! instance of "InstanceNormalization_TRT" version 1 plugin.
    //!
    TRT_DEPRECATED_API nvinfer1::IPluginV2* createInstanceNormalizationPlugin(
        float epsilon, nvinfer1::Weights scale_weights, nvinfer1::Weights bias_weights);

    //!
    //! \brief Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional
    //! namespace. The plugin library author should ensure that this function name is unique to the library. This
    //! function should be called once before accessing the Plugin Registry.
    //! \param logger Logger object to print plugin registration information
    //! \param libNamespace Namespace used to register all the plugins in this library
    //!
    TENSORRTAPI bool initLibNvInferPlugins(void* logger, char const* libNamespace);

} // extern "C"

#endif // NV_INFER_PLUGIN_H
