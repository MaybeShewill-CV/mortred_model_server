/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: RegistrationHelper.h
 * Date: 23-2-21
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_REGISTRATIONHELPER_H
#define MORTRED_MODEL_SERVER_REGISTRATIONHELPER_H

#include <memory>

#include "common/status_code.h"

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
    RegistrationHelper(const RegistrationHelper &transformer) = default;

    /***
     * constructor
     * @param transformer
     * @return
     */
    RegistrationHelper &operator=(const RegistrationHelper &transformer) = default;

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
    bool is_successfully_initialized() const;
    
  private:
    class Impl;
    std::shared_ptr<Impl> _m_pimpl;
};

}
}

#endif // MORTRED_MODEL_SERVER_REGISTRATIONHELPER_H
