#include "ash/io/http_client.hpp"

#include <curl/curl.h>

#include <array>
#include <cstddef>
#include <mutex>
#include <stop_token>
#include <string>

namespace ash {

namespace {

void ensure_curl_initialized() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

// Enough of a failed response to report what the endpoint said. A successful
// stream is not buffered -- the callback has already seen it -- so this bound
// only decides how much of an error page survives into the message.
constexpr std::size_t kErrorBodyLimit = 8 * 1024;

struct Transfer {
    const ChunkCallback* on_chunk = nullptr;
    std::string* body = nullptr;
    std::stop_token stop{};
    bool aborted = false;
};

// One write path for both calls: a streaming request has a callback and no body
// to keep, a plain one has a body and no callback, and neither branch is worth
// a second copy of the setup below it.
std::size_t write_body(char* data, std::size_t size, std::size_t count, void* user_data) {
    auto* transfer = static_cast<Transfer*>(user_data);
    const std::size_t bytes = size * count;

    if (transfer->stop.stop_requested()) {
        transfer->aborted = true;
        return 0;
    }

    if (transfer->on_chunk != nullptr) {
        bool keep_going = false;
        try {
            keep_going = (*transfer->on_chunk)(std::string_view{data, bytes});
        } catch (...) {
            // This runs inside libcurl, so an exception escaping here would
            // unwind through C. A sink that throws is a sink that has stopped
            // reading, which is a decision the transfer can act on.
            keep_going = false;
        }
        if (!keep_going) {
            transfer->aborted = true;
            return 0;
        }
    }

    if (transfer->body != nullptr && transfer->body->size() < kErrorBodyLimit) {
        transfer->body->append(data, bytes);
    }
    return bytes;
}

// libcurl calls this about once a second during a transfer, which is the
// documented way to stop one from outside. The write callback checks the token
// too, so a stream that is actively producing stops on the next chunk rather
// than on the next tick.
int abort_on_stop(void* user_data, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* transfer = static_cast<Transfer*>(user_data);
    if (transfer->stop.stop_requested()) {
        transfer->aborted = true;
        return 1;
    }
    return 0;
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

    [[nodiscard]] HttpResponse perform(const HttpRequest& request,
                                       bool is_post,
                                       const ChunkCallback* on_chunk,
                                       std::stop_token stop) const {
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
        Transfer transfer;
        transfer.on_chunk = on_chunk;
        transfer.body = &response.body;
        transfer.stop = stop;

        curl_easy_setopt(handle, CURLOPT_URL, request.url.c_str());
        curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, error_buffer.data());
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_body);
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &transfer);
        curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, request.timeout_ms);
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 15000L);
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(handle, CURLOPT_USERAGENT, "ash/0.1");
        if (stop.stop_possible()) {
            curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, abort_on_stop);
            curl_easy_setopt(handle, CURLOPT_XFERINFODATA, &transfer);
        }
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

        // Read even after a failure: the status line is parsed before anything
        // went wrong, and "aborted after a 200" is a different thing from
        // "aborted after a 500".
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &response.status);

        if (transfer.aborted) {
            response.aborted = true;
        } else if (code != CURLE_OK) {
            response.error = error_buffer[0] != '\0' ? std::string{error_buffer.data()}
                                                     : std::string{curl_easy_strerror(code)};
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

HttpResponse HttpClient::post(const HttpRequest& request) const { return impl_->perform(request, true, nullptr, {}); }

HttpResponse HttpClient::get(const HttpRequest& request) const {
    return impl_->perform(request, false, nullptr, {});
}

HttpResponse HttpClient::post_stream(const HttpRequest& request,
                                     const ChunkCallback& on_chunk,
                                     std::stop_token stop) const {
    return impl_->perform(request, true, &on_chunk, stop);
}

}  // namespace ash
