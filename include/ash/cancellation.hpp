#pragma once

#include <stdexcept>

namespace ash {

// Thrown when a stop request cuts short work that was already in flight.
//
// A cancellation is not a failure, so it does not travel as one. The agent loop
// catches this and finishes the run with stop_reason "cancelled", keeping what
// the run had produced up to that point. A caller that only ever saw
// std::runtime_error could not tell a cancelled run from a crashed one, which
// is the whole reason this is a type and not a message.
class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error{"cancelled"} {}
};

}  // namespace ash
