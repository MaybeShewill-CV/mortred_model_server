/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: mysql_db_config.cpp
 * Date: 23-2-14
 ************************************************/
 
#include "mysql_db_config.h"

#include "glog/logging.h"

#include "common/file_path_util.h"

namespace jinq {
namespace registration {

using jinq::common::FilePathUtil;
using jinq::common::StatusCode;

namespace mysql {
/***************** Impl Function Sets ******************/

class MySqlDBConfig::Impl {
  public:
    /***
     *
     */
    Impl() = default;

    /***
     *
     */
    ~Impl() = default;

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
    StatusCode init(const decltype(toml::parse(""))& config);

    /***
     * 
     * @param cfg_file_path 
     * @return 
     */
    StatusCode init(const std::string& cfg_file_path);

    /***
     *
     * @return
     */
    bool is_successfully_initialized() const {
        return _m_successfully_initialized;
    };

    /***
     *
     * @return
     */
    std::string get_user_name() const {
        return _m_user_name;
    }


    /***
     *
     * @return
     */
    std::string get_user_pw() const {
        return _m_pw;
    }

    /***
     *
     * @return
     */
    std::string get_host() const {
        return _m_host;
    }

    /***
     *
     * @return
     */
    std::string get_uri() const {
        return _m_uri;
    }

    /***
     *
     * @return
     */
    int get_port() const {
        return _m_port;
    }

    /***
     *
     * @return
     */
    std::string get_db_name() const {
        return _m_db_name;
    }

  private:
    // user name
    std::string _m_user_name;
    // password
    std::string _m_pw;
    // mysql host
    std::string _m_host;
    // mysql uri
    std::string _m_uri;
    // mysql port
    int _m_port = -1;
    // mysql db name
    std::string _m_db_name;
    // init success flag
    bool _m_successfully_initialized = false;
};

/***
*
* @param cfg_file_path
* @return
 */
StatusCode MySqlDBConfig::Impl::init(const decltype(toml::parse(""))& config) {
    if (!config.contains("MYSQL_DB")) {
        LOG(ERROR) << "config file missing MYSQL_DB field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    auto mysql_sec = config.at("MYSQL_DB");
    // init user name
    if (!mysql_sec.contains("user_name")) {
        LOG(ERROR) << "config section missing user_name field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    _m_user_name = mysql_sec.at("user_name").as_string();
    // init user password
    if (!mysql_sec.contains("password")) {
        LOG(ERROR) << "config section missing password field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    _m_pw = mysql_sec.at("password").as_string();
    // init host
    if (!mysql_sec.contains("host")) {
        LOG(ERROR) << "config section missing host field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    _m_host = mysql_sec.at("host").as_string();
    // init uri
    if (!mysql_sec.contains("uri")) {
        LOG(ERROR) << "config section missing uri field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    _m_uri = mysql_sec.at("uri").as_string();
    // init port
    if (!mysql_sec.contains("port")) {
        LOG(ERROR) << "config section missing port field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    _m_port = static_cast<int>(mysql_sec.at("port").as_integer());
    // init db_name
    if (!mysql_sec.contains("db_name")) {
        LOG(ERROR) << "config section missing db_name field";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    _m_db_name = mysql_sec.at("db_name").as_string();

    _m_successfully_initialized = true;
    return StatusCode::OJBK;
}

/***
 * 
 * @param cfg_file_path 
 * @return 
 */
StatusCode MySqlDBConfig::Impl::init(const std::string &cfg_file_path) {
    if (!FilePathUtil::is_file_exist(cfg_file_path)) {
        LOG(ERROR) << "config file: " << cfg_file_path << " not exists";
        return StatusCode::MYSQL_INIT_DB_CONFIG_ERROR;
    }
    
    auto cfg = toml::parse(cfg_file_path);
    return init(cfg);
}



/************* Export Function Sets *************/

/***
 *
 */
MySqlDBConfig::MySqlDBConfig() {
    _m_pimpl = std::make_unique<Impl>();
}

/***
 *
 * @param cfg
 * @return
 */
StatusCode MySqlDBConfig::init(const decltype(toml::parse(""))& cfg) {
    return _m_pimpl->init(cfg);
}

/***
 *
 * @param config_file_path
 * @return
 */
StatusCode MySqlDBConfig::init(const std::string &config_file_path) {
    return _m_pimpl->init(config_file_path);
}

/***
 *
 * @return
 */
bool MySqlDBConfig::is_successfully_initialized() const {
    return _m_pimpl->is_successfully_initialized();
}

/***
 *
 * @return
 */
std::string MySqlDBConfig::get_user_name() const {
    return _m_pimpl->get_user_name();
}

/***
 *
 * @return
 */
std::string MySqlDBConfig::get_user_pw() const {
    return _m_pimpl->get_user_pw();
}

/***
 *
 * @return
 */
std::string MySqlDBConfig::get_host() const {
    return _m_pimpl->get_host();
}

/***
 *
 * @return
 */
std::string MySqlDBConfig::get_uri() const {
    return _m_pimpl->get_uri();
}

/***
 *
 * @return
 */
int MySqlDBConfig::get_port() const {
    return _m_pimpl->get_port();
}

/***
 *
 * @return
 */
std::string MySqlDBConfig::get_db_name() const {
    return _m_pimpl->get_db_name();
}

}
}
}
