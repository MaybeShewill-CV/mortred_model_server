/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: RouteTable.h
 * Date: 23-2-22
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_ROUTETABLE_H
#define MORTRED_MODEL_SERVER_ROUTETABLE_H

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

#include "common/status_code.h"

namespace jinq {
namespace registration {

template<class T>
class BaseNode {
  public:
    bool contain_node(const std::string& node_name) {
        return node_map.find(node_name) != node_map.end();
    }

    void insert_node(const std::string& node_name, const T& node) {
        node_map.insert(std::make_pair(node_name, node));
    }

    std::unordered_map<std::string, T> node_map;
};

// Define the UriHandlerNode class
class UriHandlerNode {
  public:
    std::string uri;
    std::function<void()> handler;
};

// Define the VersionNode class
class VersionNode : public BaseNode<UriHandlerNode>{
  public:
    std::string version;
};

// Define the ServiceNode class
class ServiceNode : public BaseNode<VersionNode>{
  public:
    std::string service_name;
};

// Define the ProjectNode class
class ProjectNode : public BaseNode<ServiceNode>{
  public:
    std::string project_name;
};

class RootNode : public BaseNode<ProjectNode>{
  public:
    std::string root_name;
};

class RouteTable {
  public:
    /***
     *
     */
    RouteTable() {
        _m_root = std::make_unique<RootNode>();
        _m_root->root_name = "mortred_model_server";
    };

    /***
     *
     */
    ~RouteTable() = default;
    
    /***
     * constructor
     * @param transformer
     */
    RouteTable(const RouteTable &transformer) = delete;
    
    /***
     * constructor
     * @param transformer
     * @return
     */
    RouteTable &operator=(const RouteTable &transformer) = delete;

    /***
     *
     * @param proj_name
     * @param service_name
     * @param version
     * @param service_uri
     * @param handler
     * @return
     */
    jinq::common::StatusCode add_handler(
        const std::string& proj_name, const std::string& service_name,
        const std::string& version, const std::string& service_uri,
        const std::function<void()>& handler);

    /***
     *
     * @param proj_name
     * @param service_name
     * @param version
     * @param service_uri
     * @return
     */
    jinq::common::StatusCode get_handler(const std::string& proj_name, const std::string& service_name,
                                         const std::string& version, const std::string& service_uri,
                                         std::function<void()>& handler) const;

    /***
     *
     * @param project_name
     * @param service_names
     * @return
     */
    jinq::common::StatusCode list_all_service_names_of_project(const std::string& project_name, std::vector<std::string>& service_names);

    /***
     *
     * @param project_name
     * @param service_name
     * @param version
     * @param uri_names
     * @return
     */
    jinq::common::StatusCode list_all_uri_names_of_service(
        const std::string& project_name, const std::string& service_name,
        const std::string& version, std::vector<std::string>& uri_names);

  private:
    std::unique_ptr<RootNode> _m_root;
};

}
}

#endif // MORTRED_MODEL_SERVER_ROUTETABLE_H
