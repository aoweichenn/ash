#include "ash/tool/builtin.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace ash {

namespace {

namespace fs = std::filesystem;

ToolResult success(std::string content) {
    ToolResult result;
    result.content = std::move(content);
    return result;
}

ToolResult failure(std::string message) {
    ToolResult result;
    result.content = std::move(message);
    result.is_error = true;
    return result;
}

Task<ToolResult> read_file(const nlohmann::json& arguments, std::size_t max_bytes) {
    const std::string path = arguments.value("path", "");
    if (path.empty()) {
        co_return failure("read_file: 'path' is required");
    }

    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (error) {
        co_return failure("read_file: cannot stat '" + path + "': " + error.message());
    }
    if (size > max_bytes) {
        co_return failure("read_file: '" + path + "' is " + std::to_string(size) + " bytes, over the " +
                          std::to_string(max_bytes) + " byte limit");
    }

    std::ifstream input{path, std::ios::binary};
    if (!input) {
        co_return failure("read_file: cannot open '" + path + "'");
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        co_return failure("read_file: error while reading '" + path + "'");
    }
    co_return success(buffer.str());
}

Task<ToolResult> write_file(const nlohmann::json& arguments) {
    const std::string path = arguments.value("path", "");
    if (path.empty()) {
        co_return failure("write_file: 'path' is required");
    }
    const std::string content = arguments.value("content", "");

    const fs::path target{path};
    if (target.has_parent_path() && !target.parent_path().empty()) {
        std::error_code error;
        fs::create_directories(target.parent_path(), error);
        if (error) {
            co_return failure("write_file: cannot create '" + target.parent_path().string() +
                              "': " + error.message());
        }
    }

    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    if (!output) {
        co_return failure("write_file: cannot open '" + path + "' for writing");
    }
    output << content;
    output.close();
    if (!output) {
        co_return failure("write_file: failed while writing '" + path + "'");
    }
    co_return success("wrote " + std::to_string(content.size()) + " bytes to " + path);
}

Task<ToolResult> list_dir(const nlohmann::json& arguments) {
    const std::string path = arguments.value("path", ".");

    std::error_code error;
    if (!fs::exists(path, error)) {
        co_return failure("list_dir: '" + path + "' does not exist");
    }
    if (!fs::is_directory(path, error)) {
        co_return failure("list_dir: '" + path + "' is not a directory");
    }

    std::vector<std::string> entries;
    for (const auto& entry : fs::directory_iterator{path, error}) {
        std::string name = entry.path().filename().string();
        if (entry.is_directory(error)) {
            name.push_back('/');
        }
        entries.push_back(std::move(name));
    }
    if (error) {
        co_return failure("list_dir: error while reading '" + path + "': " + error.message());
    }

    std::sort(entries.begin(), entries.end());
    std::string listing;
    for (const auto& entry : entries) {
        listing += entry;
        listing.push_back('\n');
    }
    co_return success(std::move(listing));
}

}  // namespace

std::shared_ptr<Tool> make_read_file_tool(std::size_t max_bytes) {
    nlohmann::json schema = {{"type", "object"},
                             {"properties",
                              {{"path",
                                {{"type", "string"},
                                 {"description", "Path of the file to read."}}}}},
                             {"required", nlohmann::json::array({"path"})},
                             {"additionalProperties", false}};

    return make_tool("read_file", "Read a UTF-8 text file from disk.", std::move(schema),
                     [max_bytes](const nlohmann::json& arguments, std::stop_token) {
                         return read_file(arguments, max_bytes);
                     });
}

std::shared_ptr<Tool> make_write_file_tool() {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"path", {{"type", "string"}, {"description", "Path of the file to write."}}},
          {"content", {{"type", "string"}, {"description", "Full contents to write."}}}}},
        {"required", nlohmann::json::array({"path", "content"})},
        {"additionalProperties", false}};

    return make_tool("write_file", "Write a UTF-8 text file, creating parent directories.", std::move(schema),
                     [](const nlohmann::json& arguments, std::stop_token) { return write_file(arguments); });
}

std::shared_ptr<Tool> make_list_dir_tool() {
    nlohmann::json schema = {{"type", "object"},
                             {"properties",
                              {{"path",
                                {{"type", "string"},
                                 {"description", "Directory to list. Defaults to the working directory."}}}}},
                             {"additionalProperties", false}};

    return make_tool("list_dir", "List a directory's entries, directories suffixed with '/'.", std::move(schema),
                     [](const nlohmann::json& arguments, std::stop_token) { return list_dir(arguments); });
}

}  // namespace ash
