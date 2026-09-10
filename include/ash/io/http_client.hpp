#pragma once

#include <memory>
#include <string>
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

    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && status >= 200 && status < 300;
    }
};

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ash
