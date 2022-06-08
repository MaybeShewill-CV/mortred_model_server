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

namespace morted {
namespace factory {

using morted::models::BaseAiModel;

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

    void list_registered_models() {
        for (auto& model : _m_model_registry) {
            LOG(INFO) << "registered model: " << model.first << ", address: " << model.second;
        }
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

    static void list_all_models() {
        ModelFactory<BASE_AI_MODEL>::get_instance().list_registered_models();
    }
};

}
}


#endif //MM_AI_SERVER_BASE_FACTORY_H
