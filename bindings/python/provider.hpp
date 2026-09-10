#pragma once

// The model endpoint wrapper, in its own header so that run.cpp can build an
// Agent on top of it. Nothing else about it wants to be shared.

#include <memory>
#include <string>
#include <utility>

#include "ash/model/provider.hpp"

namespace ash::python {

// A model endpoint, owned.
//
// The shared_ptr is the point of this class rather than an implementation
// detail: an Agent borrows a provider for the length of a run, and holding one
// here is what lets the binding copy that ownership into the Agent instead of
// asking the caller to keep the provider alive. A Python object that went out of
// scope mid-run would otherwise leave the run pointing at freed memory, and the
// garbage collector decides when that happens.
class Provider {
public:
    explicit Provider(std::shared_ptr<ModelProvider> impl) : impl_(std::move(impl)) {}

    [[nodiscard]] const std::shared_ptr<ModelProvider>& shared() const noexcept { return impl_; }

    [[nodiscard]] std::string name() const { return std::string{impl_->name()}; }
    [[nodiscard]] std::string model() const { return impl_->model(); }

private:
    std::shared_ptr<ModelProvider> impl_;
};

}  // namespace ash::python
