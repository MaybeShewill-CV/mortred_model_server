/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: RegistrationHelper.h
 * Date: 23-2-21
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_REGISTRATIONHELPER_H
#define MORTRED_MODEL_SERVER_REGISTRATIONHELPER_H

#include <memory>

#include "workflow/WFHttpServer.h"

#include "common/status_code.h"
#include "registration/mysql/mysql_db_config.h"

namespace jinq {
namespace registration {

class RegistrationHelper {
  public:
    /***
     * constructor
     */
    RegistrationHelper() = default;
    
    /***
     *
     */
    ~RegistrationHelper() = default;
    
    /***
     * constructor
     * @param transformer
     */
    RegistrationHelper(const RegistrationHelper &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    RegistrationHelper &operator=(const RegistrationHelper &transformer) = delete;

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
    inline bool is_successfully_initialized() const {
        return _m_successfully_init;
    }
    
  private:
    bool _m_successfully_init;
    std::unique_ptr<WFHttpServer> _m_server;
    std::unique_ptr<jinq::registration::mysql::MySqlDBConfig> _m_mysql_db_cfg;
};

}
}

#endif // MORTRED_MODEL_SERVER_REGISTRATIONHELPER_H
