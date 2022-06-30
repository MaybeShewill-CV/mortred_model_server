/************************************************
* Copyright MaybeShewill-CV. All Rights Reserved.
* Author: MaybeShewill-CV
* File: base_factory.h
* Date: 22-6-7
************************************************/

#ifndef MM_AI_SERVER_BASE_FACTORY_H
#define MM_AI_SERVER_BASE_FACTORY_H

#include <memory>
#include <map>

#include "models/base_model.h"
#include "server/abstract_server.h"

namespace jinq {
namespace factory {

using jinq::models::BaseAiModel;
using jinq::server::BaseAiServer;

/***
 *
 * @tparam BASE_AI_MODEL
 */
template<class BASE_AI_MODEL>
class AiModelRegistrar {
public:
    virtual std::unique_ptr<BASE_AI_MODEL> create_model() = 0;

    AiModelRegistrar(const AiModelRegistrar& transformer) = delete;

    AiModelRegistrar& operator=(const AiModelRegistrar& transformer) = delete;

    AiModelRegistrar() = default;
    virtual ~AiModelRegistrar() = default;
};

/***
 *
 * @tparam BASE_AI_MODEL
 */
template<class BASE_AI_MODEL>
class ModelFactory {
public:

    ModelFactory(const ModelFactory& transformer) = delete;

    ModelFactory& operator=(const ModelFactory& transformer) = delete;

    static ModelFactory<BASE_AI_MODEL>& get_instance() {
        static ModelFactory<BASE_AI_MODEL> instance;
        return instance;
    }

    void register_model(AiModelRegistrar<BASE_AI_MODEL>* registrar, const std::string& name) {
        if (_m_model_registry.find(name) == _m_model_registry.end()) {
            _m_model_registry.insert(std::make_pair(name, registrar));
        } else {
            _m_model_registry[name] = registrar;
        }
    }

    std::unique_ptr<BASE_AI_MODEL> get_model(const std::string& name) {
        if (_m_model_registry.find(name) != _m_model_registry.end()) {
            auto* registry = _m_model_registry[name];
            return registry->create_model();
        }

        LOG(ERROR) << "No model named: " << name << " was found";
        return nullptr;
    }

private:
    ModelFactory() = default;
    ~ModelFactory() = default;

    std::map<std::string, AiModelRegistrar<BASE_AI_MODEL>* > _m_model_registry;
};

/***
 *
 * @tparam BASE_AI_MODEL
 * @tparam AI_MODEL
 */
template <typename BASE_AI_MODEL, typename AI_MODEL>
class ModelRegistrar : public AiModelRegistrar<BASE_AI_MODEL> {
public:
    explicit ModelRegistrar(const std::string& name) {
        ModelFactory<BASE_AI_MODEL>::get_instance().register_model(this, name);
    }

    std::unique_ptr<BASE_AI_MODEL> create_model() override {
        return std::unique_ptr<BASE_AI_MODEL>(new AI_MODEL());
    }
};

/***
 *
 * @tparam BASE_AI_SERVER
 */
template<class BASE_AI_SERVER>
class AiServerRegistrar {
public:
    virtual std::unique_ptr<BASE_AI_SERVER> create_server() = 0;

    AiServerRegistrar(const AiServerRegistrar& transformer) = delete;

    AiServerRegistrar& operator=(const AiServerRegistrar& transformer) = delete;

    AiServerRegistrar() = default;
    virtual ~AiServerRegistrar() = default;
};

/***
 *
 * @tparam BASE_AI_SERVER
 */
template<class BASE_AI_SERVER>
class ServerFactory {
public:

    ServerFactory(const ServerFactory& transformer) = delete;

    ServerFactory& operator=(const ServerFactory& transformer) = delete;

    static ServerFactory<BASE_AI_SERVER>& get_instance() {
        static ServerFactory<BASE_AI_SERVER> instance;
        return instance;
    }

    void register_server(AiServerRegistrar<BASE_AI_SERVER>* registrar, const std::string& name) {
        if (_m_server_registry.find(name) == _m_server_registry.end()) {
            _m_server_registry.insert(std::make_pair(name, registrar));
        } else {
            _m_server_registry[name] = registrar;
        }
    }

    std::unique_ptr<BASE_AI_SERVER> get_server(const std::string& name) {
        if (_m_server_registry.find(name) != _m_server_registry.end()) {
            auto* registry = _m_server_registry[name];
            return registry->create_server();
        }

        LOG(ERROR) << "No server named: " << name << " was found";
        return nullptr;
    }

private:
    ServerFactory() = default;
    ~ServerFactory() = default;

    std::map<std::string, AiServerRegistrar<BASE_AI_SERVER>* > _m_server_registry;
};

/***
 *
 * @tparam BASE_AI_MODEL
 * @tparam AI_MODEL
 */
template <typename BASE_AI_SERVER, typename AI_SERVER>
class ServerRegistrar : public AiServerRegistrar<BASE_AI_SERVER> {
public:
    explicit ServerRegistrar(const std::string& name) {
        ServerFactory<BASE_AI_SERVER>::get_instance().register_server(this, name);
    }

    std::unique_ptr<BASE_AI_SERVER> create_server() override {
        return std::unique_ptr<BASE_AI_SERVER>(new AI_SERVER());
    }
};

}
}

#endif //MM_AI_SERVER_BASE_FACTORY_H
