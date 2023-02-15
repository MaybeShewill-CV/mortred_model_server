/************************************************
 * Copyright MaybeShewill-CV. All Rights Reserved.
 * Author: MaybeShewill-CV
 * File: SqlQueryBuilder.h
 * Date: 23-2-15
 ************************************************/

#ifndef MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H
#define MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H

#include <memory>

namespace jinq {
namespace registration {
namespace mysql {

class SqlQueryBuilder {
    
  public:
    /***
     * constructor
     */
    SqlQueryBuilder();
    
    /***
     *
     */
    ~SqlQueryBuilder();
    
    /***
     * constructor
     * @param transformer
     */
    SqlQueryBuilder(const SqlQueryBuilder &transformer) = delete;
    
    /***
     * constructor
     * @param transformer
     * @return
     */
    SqlQueryBuilder &operator=(const SqlQueryBuilder &transformer) = delete;



  private:
    class Impl;
    std::unique_ptr<Impl> _m_pimpl;
};

}
}
}

#endif // MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H
