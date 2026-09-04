/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_factory.h
* Date: 22-6-7
************************************************/

#ifndef MORTRED_MODEL_SERVER_BASE_FACTORY_H
#define MORTRED_MODEL_SERVER_BASE_FACTORY_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "glog/logging.h"

namespace jinq {
namespace factory {

/***
 * Type-erased factory shared by models and servers.
 * Creator closures are owned by value (no dangling registrar pointers),
 * register/create are mutex-guarded, same-name registration overwrites.
 * @tparam BASE abstract base class
 */
template<typename BASE>
class TypeErasedFactory {
public:
    using creator_t = std::function<std::unique_ptr<BASE>()>;

    TypeErasedFactory(const TypeErasedFactory& transformer) = delete;
    TypeErasedFactory& operator=(const TypeErasedFactory& transformer) = delete;
    TypeErasedFactory(TypeErasedFactory&& transformer) = delete;
    TypeErasedFactory& operator=(TypeErasedFactory&& transformer) = delete;

    static TypeErasedFactory& get_instance() {
        static TypeErasedFactory<BASE> instance;
        return instance;
    }

    /***
     * Register an arbitrary creator closure. Used by spec-driven generic
     * servers whose concrete type is a template instantiation carrying a
     * runtime spec, not a bare default-constructible class. Same-name
     * registration overwrites; empty names and null closures are rejected.
     */
    void register_creator(const std::string& name, creator_t creator) {
        if (name.empty() || !creator) {
            LOG(ERROR) << "refusing to register a null creator or a creator with an empty name";
            return;
        }
        std::lock_guard<std::mutex> lock(_m_mutex);
        _m_creators[name] = std::move(creator);
    }

    std::unique_ptr<BASE> create(const std::string& name) const {
        creator_t creator;
        {
            std::lock_guard<std::mutex> lock(_m_mutex);
            auto iter = _m_creators.find(name);
            if (iter == _m_creators.end()) {
                LOG(ERROR) << "no type registered with name: " << name;
                return nullptr;
            }
            creator = iter->second;
        }
        return creator();
    }

private:
    TypeErasedFactory() = default;
    ~TypeErasedFactory() = default;

    mutable std::mutex _m_mutex;
    std::map<std::string, creator_t> _m_creators;
};

// server factory is the production spelling; models construct via catalogs
template<typename BASE>
using ServerFactory = TypeErasedFactory<BASE>;

}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_BASE_FACTORY_H
