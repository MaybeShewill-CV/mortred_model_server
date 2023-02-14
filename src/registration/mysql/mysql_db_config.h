/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mysql_db_config.h
 * Date: 23-2-14
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_MYSQL_DB_CONFIG_H
#define MORTRED_MODEL_SERVER_MYSQL_DB_CONFIG_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"

namespace jinq {
namespace registration {
namespace mysql {
class MySqlDBConfig {
  public:
    /***
     * constructor
     */
    MySqlDBConfig();
    
    /***
     *
     */
    ~MySqlDBConfig() = default;

    /***
     * constructor
     * @param transformer
     */
    MySqlDBConfig(const MySqlDBConfig &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    MySqlDBConfig &operator=(const MySqlDBConfig &transformer) = delete;

    /***
     *
     * @param toml
     * @return
     */
    jinq::common::StatusCode init(const decltype(toml::parse(""))& cfg);

    /***
     *
     * @param config_file_path
     * @return
     */
    jinq::common::StatusCode init(const std::string& config_file_path);

    /***
     *
     * @return
     */
    std::string get_user_name() const;


    /***
     *
     * @return
     */
    std::string get_user_pw() const;

    /***
     *
     * @return
     */
    std::string get_host() const;

    /***
     *
     * @return
     */
    std::string get_uri() const;

    /***
     *
     * @return
     */
    int get_port() const;

    /***
     *
     * @return
     */
    std::string get_db_name() const;

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const;

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
}
}
}

#endif // MORTRED_MODEL_SERVER_MYSQL_DB_CONFIG_H
