/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: VisionMcpServer.cpp
 * Date: 25-4-9
 ************************************************/

#include "vision_mcp_server.h"

#include "glog/logging.h"
#include "toml/toml.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "workflow/WFTaskFactory.h"
#include "workflow/WFHttpServer.h"

#include "common/status_code.h"
#include "common/file_path_util.h"

namespace jinq {
namespace server {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;

namespace tiny_mcp {

/************ Impl Declaration ************/

class VisionMcpServer::Impl {
  public:
    /***
    *
    * @param cfg_file_path
    * @return
     */
    StatusCode init(const decltype(toml::parse(""))& config);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() {
        return _m_successfully_initialized;
    }

  private:
    // init flag
    bool _m_successfully_initialized = false;
};

/************ Impl Implementation ************/

/***
 *
 * @param config
 * @return
 */
StatusCode VisionMcpServer::Impl::init(const decltype(toml::parse("")) &config) {

    _m_successfully_initialized = true;
    LOG(INFO) << "bisenetv2 segmentation server init successfully";
    return StatusCode::OK;
}

/***
 *
 */
VisionMcpServer::VisionMcpServer() {
    _m_impl = std::make_unique<Impl>();
}

/***
 *
 */
VisionMcpServer::~VisionMcpServer() = default;

/***
 *
 * @param cfg
 * @return
 */
StatusCode VisionMcpServer::init(const decltype(toml::parse("")) &config) {
    // init impl
    auto status = _m_impl->init(config);


    return StatusCode::OK;
}

/***
 *
 * @return
 */
bool VisionMcpServer::is_successfully_initialized() const {
    return _m_impl->is_successfully_initialized();
}

}
}
}