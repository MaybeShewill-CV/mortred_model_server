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
    SqlQueryBuilder() = default;
    
    /***
     *
     */
    ~SqlQueryBuilder() = default;
    
    /***
     * constructor
     * @param transformer
     */
    SqlQueryBuilder(const SqlQueryBuilder &transformer) = default;
    
    /***
     * constructor
     * @param transformer
     * @return
     */
    SqlQueryBuilder &operator=(const SqlQueryBuilder &transformer) = default;

    /***
     *
     * @param columns
     * @return
     */
    SqlQueryBuilder& select(const std::string& columns);

  private:
    std::string _m_query;
};

}
}
}

#endif // MORTRED_MODEL_SERVER_SQLQUERYBUILDER_H
