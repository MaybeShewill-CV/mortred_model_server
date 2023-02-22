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
    const std::function<void()> &handler) {

    if (!_m_root->contain_node(proj_name)) {
        ProjectNode proj_node;
        _m_root->insert_node(proj_name, proj_node);
    }
    auto& proj_node = _m_root->node_map[proj_name];
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
    std::function<void()> &handler) const {

    if (!_m_root->contain_node(proj_name)) {
        LOG(ERROR) << "Root table doesn\'t have project node named: " << proj_name;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& proj_node = _m_root->node_map[proj_name];
    if (!proj_node.contain_node(service_name)) {
        std::string log_str = fmt::format("Project node: {} doesn\'t have service node: {}", proj_name, service_name);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& srv_node = proj_node.node_map[service_name];
    if (!srv_node.contain_node(version)) {
        std::string log_str = fmt::format("Service node: {} doesn\'t have version node: {}", service_name, version);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& ver_node = srv_node.node_map[version];
    if (!ver_node.contain_node(service_uri)) {
        std::string log_str = fmt::format("Version node: {} doesn\'t have version node: {}", version, service_uri);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_HANDLER_FAILED;
    }
    auto& uri_handler_node = ver_node.node_map[service_uri];
    handler = uri_handler_node.handler;
    return StatusCode::OJBK;
}

/***
 *
 * @param project_name
 * @param service_name
 * @param version
 * @param uri_names
 * @return
 */
jinq::common::StatusCode RouteTable::get_all_uri_names_of_service(
    const std::string &proj_name,
    const std::string &service_name,
    const std::string &version,
    std::vector<std::string> &uri_names) {

    if (!_m_root->contain_node(proj_name)) {
        LOG(ERROR) << "Root table doesn\'t have project node named: " << proj_name;
        return StatusCode::ROUTER_GET_ALL_URI_NAMES_FAILED;
    }
    auto& proj_node = _m_root->node_map[proj_name];
    if (!proj_node.contain_node(service_name)) {
        std::string log_str = fmt::format("Project node: {} doesn\'t have service node: {}", proj_name, service_name);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_ALL_URI_NAMES_FAILED;
    }
    auto& srv_node = proj_node.node_map[service_name];
    if (!srv_node.contain_node(version)) {
        std::string log_str = fmt::format("Service node: {} doesn\'t have version node: {}", service_name, version);
        LOG(ERROR) << log_str;
        return StatusCode::ROUTER_GET_ALL_URI_NAMES_FAILED;
    }
    auto& ver_node = srv_node.node_map[version];
    for (auto& uri_handler_node : ver_node.node_map) {
        uri_names.push_back(uri_handler_node.first);
    }
    return StatusCode::OJBK;
}

/***
 *
 * @param project_name
 * @param service_names
 * @return
 */
jinq::common::StatusCode RouteTable::get_all_service_names_of_project(
    const std::string &proj_name,
    std::vector<std::string> &service_names) {

    if (!_m_root->contain_node(proj_name)) {
        LOG(ERROR) << "Root table doesn\'t have project node named: " << proj_name;
        return StatusCode::ROUTER_GET_ALL_SERVICE_NAMES_FAILED;
    }
    auto& proj_node = _m_root->node_map[proj_name];
    for (auto& iter : proj_node.node_map) {
        service_names.push_back(iter.first);
    }
    return StatusCode::OJBK;
}

}
}
