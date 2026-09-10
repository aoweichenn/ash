#pragma once

#include <cstddef>
#include <memory>

#include "ash/tool/tool.hpp"

namespace ash {

// Reads a UTF-8 text file, refusing anything larger than `max_bytes` so a
// stray binary cannot blow up the context window.
[[nodiscard]] std::shared_ptr<Tool> make_read_file_tool(std::size_t max_bytes = 262144);

// Writes a UTF-8 text file, creating parent directories as needed.
[[nodiscard]] std::shared_ptr<Tool> make_write_file_tool();

// Lists a directory's entries, one per line, directories suffixed with '/'.
[[nodiscard]] std::shared_ptr<Tool> make_list_dir_tool();

}  // namespace ash
