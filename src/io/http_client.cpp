#include "ash/io/http_client.hpp"

#include <curl/curl.h>

#include <array>
#include <cstddef>
#include <mutex>
#include <string>

namespace ash {

namespace {

void ensure_curl_initialized() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

std::size_t write_body(char* data, std::size_t size, std::size_t count, void* user_data) {
    const std::size_t bytes = size * count;
    static_cast<std::string*>(user_data)->append(data, bytes);
    return bytes;
}

void lock_share(CURL* /*handle*/, curl_lock_data /*data*/, curl_lock_access /*access*/, void* user_ptr) {
    static_cast<std::mutex*>(user_ptr)->lock();
}

void unlock_share(CURL* /*handle*/, curl_lock_data /*data*/, void* user_ptr) {
    static_cast<std::mutex*>(user_ptr)->unlock();
}

}  // namespace

struct HttpClient::Impl {
    CURLSH* share = nullptr;
    std::mutex share_mutex;

    Impl() {
        ensure_curl_initialized();
        share = curl_share_init();
        if (share != nullptr) {
            curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
            curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_SSL_SESSION);
            curl_share_setopt(share, CURLSHOPT_LOCKFUNC, lock_share);
            curl_share_setopt(share, CURLSHOPT_UNLOCKFUNC, unlock_share);
            curl_share_setopt(share, CURLSHOPT_USERDATA, &share_mutex);
        }
    }

    ~Impl() {
        if (share != nullptr) {
            curl_share_cleanup(share);
        }
    }

    [[nodiscard]] HttpResponse perform(const HttpRequest& request, bool is_post) const {
        HttpResponse response;

        CURL* handle = curl_easy_init();
        if (handle == nullptr) {
            response.error = "curl_easy_init failed";
            return response;
        }

        curl_slist* header_list = nullptr;
        for (const auto& header : request.headers) {
            header_list = curl_slist_append(header_list, header.c_str());
        }

        std::array<char, CURL_ERROR_SIZE> error_buffer{};

        curl_easy_setopt(handle, CURLOPT_URL, request.url.c_str());
        curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, error_buffer.data());
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_body);
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response.body);
        curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, request.timeout_ms);
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 15000L);
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(handle, CURLOPT_USERAGENT, "ash/0.1");
        if (header_list != nullptr) {
            curl_easy_setopt(handle, CURLOPT_HTTPHEADER, header_list);
        }
        if (share != nullptr) {
            curl_easy_setopt(handle, CURLOPT_SHARE, share);
        }

        if (is_post) {
            curl_easy_setopt(handle, CURLOPT_POST, 1L);
            curl_easy_setopt(handle, CURLOPT_POSTFIELDS, request.body.c_str());
            curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE, static_cast<long>(request.body.size()));
        }

        const CURLcode code = curl_easy_perform(handle);
        if (code != CURLE_OK) {
            response.error = error_buffer[0] != '\0' ? std::string{error_buffer.data()}
                                                     : std::string{curl_easy_strerror(code)};
        } else {
            curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &response.status);
        }

        if (header_list != nullptr) {
            curl_slist_free_all(header_list);
        }
        curl_easy_cleanup(handle);
        return response;
    }
};

HttpClient::HttpClient() : impl_(std::make_unique<Impl>()) {}

HttpClient::~HttpClient() = default;

HttpResponse HttpClient::post(const HttpRequest& request) const { return impl_->perform(request, true); }

HttpResponse HttpClient::get(const HttpRequest& request) const { return impl_->perform(request, false); }

}  // namespace ash
