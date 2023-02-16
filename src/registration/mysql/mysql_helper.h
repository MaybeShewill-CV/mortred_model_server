/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: MySqlHelper.h
 * Date: 23-2-14
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_MYSQLHELPER_H
#define MORTRED_MODEL_SERVER_MYSQLHELPER_H

#include <memory>

#include "toml/toml.hpp"

#include "common/status_code.h"
#include "registration/mysql/mysql_db_config.h"
#include "registration/mysql/mysql_data_type.h"

namespace jinq {
namespace registration {
namespace mysql {

class MySqlHelper {
  public:
    /***
     * constructor
     */
    MySqlHelper();

    /***
     *
     * @param cfg
     */
    explicit MySqlHelper(const jinq::registration::mysql::MySqlDBConfig& cfg);
    
    /***
     *
     */
    ~MySqlHelper();
    
    /***
     * constructor
     * @param transformer
     */
    MySqlHelper(const MySqlHelper &transformer) = delete;

    /***
     * constructor
     * @param transformer
     * @return
     */
    MySqlHelper &operator=(const MySqlHelper &transformer) = delete;
    
    /***
     *
     * @return
     */
    bool is_successfully_initialized() const;

    /***
     *
     * @param db_cfg
     * @return
     */
    jinq::common::StatusCode init(const jinq::registration::mysql::MySqlDBConfig& db_cfg);

    /***
     *
     * @param table
     * @param columns
     * @param conditions
     * @return
     */
    std::string select(
        const std::string& table,
        const std::vector<std::string>& columns,
        const std::map<std::string, std::string>& conditions);

  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};
}

}
}

#endif // MORTRED_MODEL_SERVER_MYSQLHELPER_H
