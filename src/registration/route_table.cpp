/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: RouteTable.cpp
 * Date: 23-2-22
 ************************************************/

#include "route_table.h"

#include "glog/logging.h"
#include "fmt/core.h"

namespace jinq {
namespace registration {

using protocol::HttpRequest;
using protocol::HttpResponse;
using jinq::common::StatusCode;

/***
 *
 * @param proj_name
 * @param service_name
 * @param version
 * @param service_uri
 * @param handler
 * @return
 */
jinq::common::StatusCode RouteTable::add_handler(
    const std::string &proj_name, const std::string &service_name,
    const std::string &version, const std::string &service_uri,
    const std::function<void(const HttpRequest*, HttpResponse*)> &handler) {

    if (!_m_root.contain_node(proj_name)) {
        ProjectNode proj_node;
        _m_root.insert_node(proj_name, proj_node);
    }
    auto& proj_node = _m_root.node_map[proj_name];
    if (!proj_node.contain_node(service_name)) {
        ServiceNode sv_node;
        proj_node.insert_node(service_name, sv_node);
    }
    auto& sv_node = proj_node.node_map[service_name];
    if (!sv_node.contain_node(version)) {
        VersionNode version_node;
        sv_node.insert_node(version, version_node);
    }
    auto& ver_node = sv_node.node_map[version];
    if (!ver_node.contain_node(service_uri)) {
        UriHandlerNode uri_handle_node;
        uri_handle_node.uri = service_uri;
        uri_handle_node.handler = handler;
        ver_node.insert_node(service_uri, uri_handle_node);
    } else {
        std::string log_info = fmt::format("handler for uri {}/{}/{}/{} already exists", proj_name, service_name, version, service_uri);
        LOG(ERROR) << log_info;
        return StatusCode::ROUTER_ADD_HANDLER_FAILED;
    }

    return StatusCode::OK;
}

/***
 *
 * @param uri
 * @param handler
 * @return
 */
jinq::common::StatusCode RouteTable::add_handler(
    const std::string &uri,
    const std::function<void(const HttpRequest*, HttpResponse*)> &handler) {
    // split uri into /root/project/service/version/uri
    std::string temp;
    std::stringstream ss { uri };
    std::vector<std::string> uri_split;
    while (std::getline(ss, temp, '/')) {
        uri_split.push_back(temp);
    }

    if (uri_split.size() != 5) {
        std::string log_str = fmt::format(
            "uri: {} has wrong format which should be eg. /mmai_server/project/service/version/uri", uri);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_ADD_HANDLER_FAILED;
    }

    std::string proj = uri_split[0];
    std::string service = uri_split[0];
    std::string version = uri_split[0];
    std::string uri_name = uri_split[0];

    return add_handler(proj, service, version, uri_name, handler);
}

/***
 *
 * @param proj_name
 * @param service_name
 * @param version
 * @param service_uri
 * @param handler
 * @return
 */
jinq::common::StatusCode RouteTable::get_handler(
    const std::string &proj_name, const std::string &service_name,
    const std::string &version, const std::string &service_uri,
    std::function<void(const HttpRequest*, HttpResponse*)> &handler) const {

    if (!_m_root.contain_node(proj_name)) {
        LOG(ERROR) << "Root table doesn\'t have project node named: " << proj_name;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& proj_node = _m_root.node_map.find(proj_name)->second;
    if (!proj_node.contain_node(service_name)) {
        std::string log_str = fmt::format("Project node: {} doesn\'t have service node: {}", proj_name, service_name);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& srv_node = proj_node.node_map.find(service_name)->second;
    if (!srv_node.contain_node(version)) {
        std::string log_str = fmt::format("Service node: {} doesn\'t have version node: {}", service_name, version);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& ver_node = srv_node.node_map.find(version)->second;
    if (!ver_node.contain_node(service_uri)) {
        std::string log_str = fmt::format("Version node: {} doesn\'t have version node: {}", version, service_uri);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& uri_handler_node = ver_node.node_map.find(service_uri)->second;
    handler = uri_handler_node.handler;
    return StatusCode::OJBK;
}

/***
 *
 * @param uri
 * @param handler
 * @return
 */
jinq::common::StatusCode RouteTable::get_handler(
    const std::string &uri,
    std::function<void(const HttpRequest*, HttpResponse*)> &handler) const {
    // split uri into /root/project/service/version/uri
    std::string temp;
    std::stringstream ss { uri };
    std::vector<std::string> uri_split;
    while (std::getline(ss, temp, '/')) {
        uri_split.push_back(temp);
    }

    if (uri_split.size() != 5) {
        std::string log_str = fmt::format(
            "uri: {} has wrong format which should be eg. /mmai_server/project/service/version/uri", uri);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }

    std::string proj = uri_split[0];
    std::string service = uri_split[0];
    std::string version = uri_split[0];
    std::string uri_name = uri_split[0];

    return get_handler(proj, service, version, uri_name, handler);
}

}
}
