#pragma once

#include <cstddef>
#include <memory>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "ash/io/async_semaphore.hpp"
#include "ash/model/provider.hpp"
#include "ash/model/stream.hpp"
#include "ash/task.hpp"

namespace ash {

// Caps how many requests are in flight against one provider at a time.
//
// Endpoints enforce their own concurrency limits and answer a burst with 429s
// or a dropped connection, so the limit belongs to the provider rather than to
// whoever happens to be calling it. Putting it here means a caller can fan out
// as widely as it likes -- a hundred tasks, a hundred threads -- and the
// provider still only ever sees `max_inflight` of them at once.
//
// A decorator, like the recorder: the agent loop and the fan-out code cannot
// tell whether a limiter is in the way.
class LimitedProvider : public ModelProvider {
public:
    // Zero means unlimited, which is expressed by not wrapping at all. Passing
    // it here would be a limiter that permits nothing, so it is rejected.
    LimitedProvider(std::unique_ptr<ModelProvider> inner, std::size_t max_inflight)
        : inner_(std::move(inner)), permits_(max_inflight), limit_(max_inflight) {
        if (inner_ == nullptr) {
            throw std::invalid_argument{"LimitedProvider needs a provider to wrap"};
        }
        if (max_inflight == 0) {
            throw std::invalid_argument{"LimitedProvider needs a limit above zero"};
        }
    }

    [[nodiscard]] std::string_view name() const noexcept override { return inner_->name(); }

    [[nodiscard]] const std::string& model() const noexcept override { return inner_->model(); }

    [[nodiscard]] std::size_t max_inflight() const noexcept { return limit_; }

    Task<ChatResponse> chat(ChatRequest request) override {
        // Acquired before the call and released after it, whatever it does --
        // including throwing, which is why the permit is given back on the way
        // out of the catch rather than only on the happy path. A leaked permit
        // shrinks the limit permanently, and the symptom would be a suite that
        // gets slower and slower for no visible reason.
        co_await permits_.acquire();

        try {
            ChatResponse response = co_await inner_->chat(std::move(request));
            permits_.release();
            co_return response;
        } catch (...) {
            permits_.release();
            throw;
        }
    }

    // A streamed call holds a permit for as long as the stream is open, which is
    // the whole point: the limit is about how many requests the endpoint is
    // serving, and a request does not stop counting because its answer arrives
    // in pieces.
    Task<ChatResponse> chat_stream(ChatRequest request, StreamSink& sink, std::stop_token stop = {}) override {
        co_await permits_.acquire();

        try {
            ChatResponse response = co_await inner_->chat_stream(std::move(request), sink, stop);
            permits_.release();
            co_return response;
        } catch (...) {
            permits_.release();
            throw;
        }
    }

private:
    std::unique_ptr<ModelProvider> inner_;
    AsyncSemaphore permits_;
    std::size_t limit_;
};

}  // namespace ash
