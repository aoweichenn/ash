#pragma once

#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace ash {

struct HttpRequest {
    std::string url;
    std::vector<std::string> headers;  // each entry is "Name: value"
    std::string body;
    long timeout_ms = 120000;
};

struct HttpResponse {
    long status = 0;
    std::string body;
    std::string error;  // set only on a transport-level failure

    // The transfer was cut short on purpose: either the stop token fired or the
    // chunk callback asked to stop reading. Kept apart from `error` because a
    // cancelled request did not fail -- it stopped, and a caller that cannot
    // tell those apart will report a cancellation as an outage.
    bool aborted = false;

    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && !aborted && status >= 200 && status < 300;
    }
};

// Handed each piece of the response body as it arrives. Returning false stops
// the transfer, which is how a consumer that has read the end of the stream
// avoids waiting for a connection the server has not closed.
using ChunkCallback = std::function<bool(std::string_view)>;

// Thin wrapper over libcurl. One easy handle per in-flight request, with DNS
// and TLS session state shared between them.
class HttpClient {
public:
    HttpClient();
    ~HttpClient();

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    [[nodiscard]] HttpResponse post(const HttpRequest& request) const;
    [[nodiscard]] HttpResponse get(const HttpRequest& request) const;

    // The same POST, with the body handed over as it arrives instead of being
    // accumulated. Non-2xx responses are not streamed to the callback's
    // consumer in any meaningful way -- what arrives is an error body, not a
    // stream -- so a bounded prefix of the body is kept regardless, and that is
    // what a caller reports when the request fails.
    [[nodiscard]] HttpResponse post_stream(const HttpRequest& request,
                                           const ChunkCallback& on_chunk,
                                           std::stop_token stop = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ash
