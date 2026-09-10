#include "ash/tool/shell.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

namespace ash {

namespace {

ToolResult failure(std::string message) {
    ToolResult result;
    result.content = std::move(message);
    result.is_error = true;
    return result;
}

#if defined(__unix__) || defined(__APPLE__)

ToolResult success(std::string content) {
    ToolResult result;
    result.content = std::move(content);
    return result;
}

using Clock = std::chrono::steady_clock;

// How long poll waits before the loop looks at the stop token and the deadline
// again. This is the whole of the stop latency, which is why there is no
// self-pipe anywhere in here.
constexpr std::chrono::milliseconds kSlice{100};

// How long a command gets to die after SIGTERM before it is killed outright.
constexpr std::chrono::milliseconds kKillGrace{2000};

// How long the output is allowed to keep trickling in after the command itself
// has been reaped.
constexpr std::chrono::milliseconds kDrainGrace{200};

// Closes a descriptor on every path out of this file, including the ones that
// leave through co_return.
class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) noexcept : fd_(fd) {}
    ~Fd() { reset(); }

    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;

    Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

    Fd& operator=(Fd&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return fd_; }

    void reset(int fd = -1) noexcept {
        if (fd_ >= 0) {
            (void)::close(fd_);
        }
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

// What the command said, and whether any of it had to be dropped.
struct Capture {
    std::string output;
    std::size_t total = 0;
    bool truncated = false;
    bool eof = false;
};

// Reads whatever is there right now and appends it.
//
// The cap is on what is kept, never on what is read: a child that fills the
// pipe and finds nobody reading blocks forever, so output past the limit is
// counted and thrown away rather than left where it will wedge the command.
void pump(int fd, Capture& capture, std::size_t max_bytes) {
    char buffer[8192];
    for (;;) {
        const ssize_t count = ::read(fd, buffer, sizeof(buffer));
        if (count > 0) {
            const auto read_bytes = static_cast<std::size_t>(count);
            capture.total += read_bytes;
            const std::size_t room =
                capture.output.size() < max_bytes ? max_bytes - capture.output.size() : 0;
            const std::size_t keep = std::min(room, read_bytes);
            capture.output.append(buffer, keep);
            capture.truncated = capture.truncated || keep < read_bytes;
            continue;
        }
        if (count == 0) {
            capture.eof = true;
            return;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return;  // Drained for now; the next poll says when there is more.
        }
        // A read error ends the output. The exit status is what reports it.
        capture.eof = true;
        return;
    }
}

// True when there may be something to read within `slice`.
bool wait_for_output(int fd, std::chrono::milliseconds slice) {
    struct pollfd entry{};
    entry.fd = fd;
    entry.events = POLLIN;
    return ::poll(&entry, 1, static_cast<int>(slice.count())) > 0;
}

// Separates a note from output that did not end in a newline.
void append_note(std::string& content, const std::string& note) {
    if (!content.empty() && content.back() != '\n') {
        content.push_back('\n');
    }
    content += note;
}

Task<ToolResult> run_shell(const nlohmann::json& arguments, std::stop_token stop,
                           const ShellOptions& options) {
    // find() rather than value(): value() throws a type error when the model
    // sends a number where a string belongs, and a tool that throws unwinds
    // straight out of the agent loop.
    const auto command_field = arguments.find("command");
    if (command_field == arguments.end() || !command_field->is_string()) {
        co_return failure("run_shell: 'command' is required and must be a string");
    }
    const std::string command = command_field->get<std::string>();
    if (command.empty()) {
        co_return failure("run_shell: 'command' is required");
    }
    if (stop.stop_requested()) {
        co_return failure("run_shell: cancelled before the command started");
    }

    int ends[2] = {-1, -1};
    if (::pipe(ends) != 0) {
        co_return failure("run_shell: cannot create a pipe: " + std::string(std::strerror(errno)));
    }
    Fd read_end{ends[0]};
    Fd write_end{ends[1]};

    posix_spawn_file_actions_t actions;
    ::posix_spawn_file_actions_init(&actions);
    // An empty stdin, so a command that reads cannot take the caller's input --
    // in a session that input is the user's next line.
    ::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    ::posix_spawn_file_actions_adddup2(&actions, write_end.get(), STDOUT_FILENO);
    ::posix_spawn_file_actions_adddup2(&actions, write_end.get(), STDERR_FILENO);
    ::posix_spawn_file_actions_addclose(&actions, read_end.get());
    ::posix_spawn_file_actions_addclose(&actions, write_end.get());

    posix_spawnattr_t attributes;
    ::posix_spawnattr_init(&attributes);
    short flags = POSIX_SPAWN_SETPGROUP;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    // glibc's way of saying only the descriptors named above survive into the
    // child. Elsewhere the child inherits the rest, which is untidy but not
    // wrong.
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    ::posix_spawnattr_setflags(&attributes, flags);
    // Its own process group, so a command that spawns children can be stopped
    // as a tree instead of leaving them running.
    ::posix_spawnattr_setpgroup(&attributes, 0);

    char dash_c[] = "-c";
    char* const argv[] = {const_cast<char*>(options.shell.c_str()), dash_c,
                          const_cast<char*>(command.c_str()), nullptr};

    pid_t pid = -1;
    const int spawned =
        ::posix_spawn(&pid, options.shell.c_str(), &actions, &attributes, argv, environ);
    ::posix_spawn_file_actions_destroy(&actions);
    ::posix_spawnattr_destroy(&attributes);

    // Let go of the write end. Holding it would mean the read end never sees
    // EOF and every command would run until its timeout.
    write_end.reset();

    if (spawned != 0) {
        co_return failure("run_shell: cannot run '" + options.shell +
                          "': " + std::string(std::strerror(spawned)));
    }

    // Non-blocking, so draining can stop when the pipe runs dry instead of
    // waiting inside read() for something that may never come.
    const int descriptor_flags = ::fcntl(read_end.get(), F_GETFL, 0);
    if (descriptor_flags < 0 ||
        ::fcntl(read_end.get(), F_SETFL, descriptor_flags | O_NONBLOCK) != 0) {
        (void)::kill(-pid, SIGKILL);
        (void)::waitpid(pid, nullptr, 0);
        co_return failure("run_shell: cannot make the output pipe non-blocking: " +
                          std::string(std::strerror(errno)));
    }

    const auto deadline = Clock::now() + options.timeout;

    Capture capture;
    int status = 0;
    bool reaped = false;
    bool timed_out = false;
    bool cancelled = false;
    bool signalled = false;
    Clock::time_point signalled_at{};

    for (;;) {
        // Reaped before the stop check, so a command that has already finished
        // is reported by how it finished rather than as a cancellation.
        int wait_status = 0;
        const pid_t waited = ::waitpid(pid, &wait_status, WNOHANG);
        if (waited == pid) {
            status = wait_status;
            reaped = true;
            break;
        }
        if (waited < 0 && errno != EINTR) {
            break;
        }

        const auto now = Clock::now();
        if (!signalled) {
            if (stop.stop_requested()) {
                cancelled = true;
            } else if (now >= deadline) {
                timed_out = true;
            }
            if (cancelled || timed_out) {
                signalled = true;
                signalled_at = now;
                // The group, not the shell: `sh -c "a | b"` would otherwise
                // leave b running with the pipe still open.
                (void)::kill(-pid, SIGTERM);
            }
        } else if (now - signalled_at >= kKillGrace) {
            // Repeats until the child is reaped. A process group cannot be
            // recycled while the shell is still a zombie in it, so this cannot
            // land on somebody else's group.
            (void)::kill(-pid, SIGKILL);
        }

        if (capture.eof) {
            // Nothing more can arrive, so the only thing left to wait for is
            // the command itself.
            (void)::poll(nullptr, 0, static_cast<int>(kSlice.count()));
            continue;
        }

        struct pollfd entry{};
        entry.fd = read_end.get();
        entry.events = POLLIN;
        const int ready = ::poll(&entry, 1, static_cast<int>(kSlice.count()));
        if (ready > 0) {
            pump(read_end.get(), capture, options.max_output_bytes);
        } else if (ready < 0 && errno != EINTR) {
            // The pipe is unusable; the exit status is all that is left.
            capture.eof = true;
        }
    }

    // The command is gone, but what it wrote may still be in the pipe, and
    // anything it started still holds the write end open. `sh -c "sleep 9 &"`
    // exits at once and yet the pipe stays open until the sleep ends, which is
    // the one way this could sit here until its timeout.
    const auto quiet_deadline = Clock::now() + kDrainGrace;
    while (!capture.eof && Clock::now() < quiet_deadline) {
        if (wait_for_output(read_end.get(), kSlice)) {
            pump(read_end.get(), capture, options.max_output_bytes);
        }
    }
    if (!capture.eof) {
        // Something outlived the command and is holding the pipe. Take the
        // group down; the output already collected is kept either way.
        (void)::kill(-pid, SIGKILL);
        const auto final_deadline = Clock::now() + kDrainGrace;
        while (!capture.eof && Clock::now() < final_deadline) {
            if (wait_for_output(read_end.get(), kSlice)) {
                pump(read_end.get(), capture, options.max_output_bytes);
            }
        }
    }

    std::string content = std::move(capture.output);
    if (capture.truncated) {
        append_note(content, "[output truncated at " + std::to_string(options.max_output_bytes) +
                                 " bytes; " + std::to_string(capture.total) + " bytes total]");
    }
    if (cancelled) {
        append_note(content, "[command cancelled]");
    } else if (timed_out) {
        append_note(content, "[command timed out after " +
                                 std::to_string(options.timeout.count()) + " ms and was killed]");
    }

    if (cancelled || timed_out) {
        co_return failure(std::move(content));
    }
    if (!reaped) {
        append_note(content, "[lost track of the command's exit status]");
        co_return failure(std::move(content));
    }
    if (WIFSIGNALED(status)) {
        append_note(content, "[killed by signal " + std::to_string(WTERMSIG(status)) + "]");
        co_return failure(std::move(content));
    }
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (code != 0) {
        append_note(content, "[exit status " + std::to_string(code) + "]");
        co_return failure(std::move(content));
    }
    co_return success(std::move(content));
}

#else

Task<ToolResult> run_shell(const nlohmann::json&, std::stop_token, const ShellOptions&) {
    co_return failure("run_shell is not supported on this platform");
}

#endif

}  // namespace

std::shared_ptr<Tool> make_shell_tool(ShellOptions options) {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"command",
           {{"type", "string"},
            {"description", "Shell command line to run. It goes through a shell, so pipes, "
                            "redirection and && work. Its stdin is empty."}}}}},
        {"required", nlohmann::json::array({"command"})},
        {"additionalProperties", false}};

    return make_tool(
        "run_shell",
        "Run a shell command and return its combined output. Use it to build, test and search. "
        "Prefer read_file and list_dir when you only need to look at something: their output is "
        "exact and is never truncated or given a timeout.",
        std::move(schema),
        [options](const nlohmann::json& arguments, std::stop_token stop) {
            return run_shell(arguments, stop, options);
        });
}

}  // namespace ash
