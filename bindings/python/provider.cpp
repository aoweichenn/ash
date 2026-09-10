#include "provider.hpp"

#include <memory>
#include <optional>
#include <string>

#include "ash/model/providers.hpp"
#include "types.hpp"

namespace ash::python {

namespace {

Provider make_provider(bool anthropic,
                       const std::string& base_url,
                       const std::string& api_key,
                       const std::string& model,
                       const std::optional<int>& max_tokens,
                       long timeout_ms) {
    ProviderConfig config;
    config.base_url = base_url;
    config.api_key = api_key;
    config.model = model;
    config.max_tokens = max_tokens;
    config.timeout_ms = timeout_ms;

    // The config is dropped here and never stored: the key lives in the
    // provider's own request headers and nowhere else, so nothing Python can
    // reach would print it in a repr or a traceback.
    return Provider{anthropic ? make_anthropic(config) : make_openai_compatible(config)};
}

}  // namespace

void register_provider(py::module_& m) {
    py::class_<Provider>(m, "Provider", "A model endpoint. Construct one with a factory below.")
        .def_property_readonly("name", &Provider::name)
        .def_property_readonly("model", &Provider::model)
        .def("__repr__", [](const Provider& provider) {
            return "Provider(" + provider.name() + " / " + provider.model() + ")";
        });

    // No default base_url on either factory. A project whose headline is that a
    // run can be reproduced offline should not have a default that quietly
    // reaches the public internet when a test forgets an argument.
    m.def(
        "openai_compatible",
        [](const std::string& base_url, const std::string& api_key, const std::string& model,
           const std::optional<int>& max_tokens, long timeout_ms) {
            return make_provider(false, base_url, api_key, model, max_tokens, timeout_ms);
        },
        py::arg("base_url"), py::arg("api_key"), py::arg("model"),
        py::arg("max_tokens") = py::none(), py::arg("timeout_ms") = 120000,
        "Speaks the /chat/completions dialect: OpenAI, DeepSeek, Moonshot, Qwen, and most other "
        "hosted endpoints.");

    m.def(
        "anthropic",
        [](const std::string& base_url, const std::string& api_key, const std::string& model,
           const std::optional<int>& max_tokens, long timeout_ms) {
            return make_provider(true, base_url, api_key, model, max_tokens, timeout_ms);
        },
        py::arg("base_url"), py::arg("api_key"), py::arg("model"),
        py::arg("max_tokens") = py::none(), py::arg("timeout_ms") = 120000,
        "Speaks the Anthropic Messages dialect.");
}

}  // namespace ash::python
