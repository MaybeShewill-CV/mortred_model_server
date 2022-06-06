/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: DBTextDetector.cpp
* Date: 22-6-6
************************************************/

#include "db_text_detector.h"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "MNN/Interpreter.hpp"

#include "common/file_path_util.h"

namespace morted {
namespace models {
using morted::common::FilePathUtil;
using morted::common::StatusCode;

namespace image_ocr {

/***************** Impl Function Sets ******************/

class DBTextDetector::Impl {
public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() {
        if (_m_net != nullptr && _m_session != nullptr) {
            _m_net->releaseModel();
            _m_net->releaseSession(_m_session);
        }
    }

    /***
    *
    * @param transformer
    */
    Impl(const Impl& transformer) = delete;

    /***
     *
     * @param transformer
     * @return
     */
    Impl& operator=(const Impl& transformer) = delete;

    /***
     *
     * @param cfg_file_path
     * @return
     */
    StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

    /***
     *
     */
    inline void release_resource() {
        if (_m_session != nullptr && _m_net != nullptr) {
            _m_net->releaseModel();
            _m_net->releaseSession(_m_session);
        }

        _m_successfully_initialized = false;
    }

public:
    // 模型文件存储路径
    std::string _m_model_file_path;
    // MNN Interpreter
    std::unique_ptr<MNN::Interpreter> _m_net = nullptr;
    // MNN Session
    MNN::Session* _m_session = nullptr;
    // MNN Input tensor node
    MNN::Tensor* _m_input_tensor = nullptr;
    // MNN Output tensor node
    MNN::Tensor* _m_output_tensor = nullptr;
    // MNN后端使用线程数
    int _m_threads_nums = 4;
    // 得分阈值
    double _m_score_threshold = 0.4;
    // nms阈值
    double _m_nms_threshold = 0.35;
    // rotate bbox 短边阈值
    float _m_sside_threshold = 3;
    // top_k keep阈值
    long _m_keep_topk = 250;
    // 用户输入网络的图像尺寸
    cv::Size _m_input_size_user = cv::Size();
    //　计算图定义的输入node尺寸
    cv::Size _m_input_size_host = cv::Size();
    // segmentation prob mat
    cv::Mat _m_seg_prob_mat;
    // segmentation score map
    cv::Mat _m_seg_score_mat;
    // 是否成功初始化标志位
    bool _m_successfully_initialized = false;

public:
    /***
     * 图像预处理, 转换图像为CV_32FC3, 通过dst = src / 127.5 - 1.0来归一化图像到[-1.0, 1.0]
     * @param input_image : 输入图像
     */
    cv::Mat preprocess_image(const cv::Mat& input_image) const;

    /***
     *
     * @param input_image_block
     */
    std::vector<dbtext_output> parse_image_blocks(const cv::Mat& input_image_block) const;

    /***
     *
     */
    std::vector<dbtext_output> postprocess_image_blocks() const;

    /***
     *
     * @return
     */
    void decode_segmentation_result_mat() const;

    /***
     *
     * @param seg_probs_mat
     * @return
     */
    std::vector<dbtext_output> get_boxes_from_bitmap() const;
};


/***
*
* @param cfg_file_path
* @return
*/
StatusCode DBTextDetector::Impl::init(const decltype(toml::parse(""))& cfg) {

    _m_successfully_initialized = true;
    LOG(INFO) << "DB_Text detection model: " << FilePathUtil::get_file_name(_m_model_file_path)
              << " initialization complete!!!";
    return StatusCode::OK;
}

/************* Export Function Sets *************/

/***
*
*/
DBTextDetector::~DBTextDetector() = default;

/***
*
* @param cfg_file_path
* @return
*/
StatusCode DBTextDetector::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
*
* @param in_out
* @return
*/
template<class INPUT, class OUTPUT>
typename std::enable_if <
std::is_same<INPUT, dbtext_input>::value&& std::is_same<OUTPUT, dbtext_output>::value,
    morted::common::StatusCode >::type
DBTextDetector::run(const INPUT* input, OUTPUT* output) {
    LOG(INFO) << "run is same dbtext input";
    return common::StatusCode::OK;
}

/***
*
* @param in_out
* @return
*/
//template<class INPUT, class OUTPUT>
//typename std::enable_if < !std::is_same<INPUT, dbtext_input>::value, StatusCode >::type
//DBTextDetector::run(const INPUT* input, OUTPUT* output) {
//    LOG(INFO) << "run is same common input";
//    return common::StatusCode::OK;
//}

template<class INPUT, class OUTPUT>
typename std::enable_if<std::is_same<INPUT, std::string>::value, morted::common::StatusCode>::type
run(const INPUT* input, OUTPUT* output) {
    LOG(INFO) << "run is same string input";
    return common::StatusCode::OK;
}

/***
*
* @return
*/
bool DBTextDetector::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}


DBTextDetector::DBTextDetector() {
    _m_pimpl = std::make_unique<Impl>();
}

}
}
}