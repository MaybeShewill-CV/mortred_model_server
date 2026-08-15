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
#include <type_traits>

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
    TypeErasedFactory(const TypeErasedFactory& transformer) = delete;
    TypeErasedFactory& operator=(const TypeErasedFactory& transformer) = delete;
    TypeErasedFactory(TypeErasedFactory&& transformer) = delete;
    TypeErasedFactory& operator=(TypeErasedFactory&& transformer) = delete;

    static TypeErasedFactory& get_instance() {
        static TypeErasedFactory<BASE> instance;
        return instance;
    }

    template<typename CONCRETE>
    void register_type(const std::string& name) {
        static_assert(std::is_base_of<BASE, CONCRETE>::value,
                      "TypeErasedFactory: CONCRETE must derive from BASE");
        if (name.empty()) {
            LOG(ERROR) << "refusing to register a creator with an empty name";
            return;
        }
        std::lock_guard<std::mutex> lock(_m_mutex);
        _m_creators[name] = []() -> std::unique_ptr<BASE> {
            return std::unique_ptr<BASE>(new CONCRETE());
        };
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

    using creator_t = std::function<std::unique_ptr<BASE>()>;

    mutable std::mutex _m_mutex;
    std::map<std::string, creator_t> _m_creators;
};

// model and server factories share this implementation; only the base differs
template<typename BASE>
using ModelFactory = TypeErasedFactory<BASE>;

template<typename BASE>
using ServerFactory = TypeErasedFactory<BASE>;

}  // namespace factory
}  // namespace jinq

#endif //MORTRED_MODEL_SERVER_BASE_FACTORY_H
