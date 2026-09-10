#include "types.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>

#include <pybind11/stl/filesystem.h>

#include "ash_eval.hpp"
#include "report.hpp"

namespace ash::python {

namespace {

namespace eval = ash::eval;

py::list jobs_of(const eval::Suite& suite) {
    py::list jobs;
    for (const eval::Job& job : suite.jobs) {
        py::dict entry;
        entry["id"] = job.id;
        entry["task"] = job.task;
        entry["journal"] = job.journal.string();
        entry["actor"] = job.actor;
        jobs.append(std::move(entry));
    }
    return jobs;
}

// Reports come back with kind already spelled out, because a caller writing
// `if delta["kind"] == "regressed"` is the whole reason the enum exists -- and
// the spelling lives in to_string(), which is what the CLI's diff prints.
py::list deltas_of(const eval::ReportDiff& diff) {
    py::list deltas;
    for (const eval::JobDelta& delta : diff.jobs) {
        py::dict entry;
        entry["id"] = delta.id;
        entry["kind"] = std::string{eval::to_string(delta.kind)};
        entry["notes"] = delta.notes;
        deltas.append(std::move(entry));
    }
    return deltas;
}

std::string describe_report(const eval::SuiteReport& report) {
    return "SuiteReport(suite=" + py::repr(py::str(report.suite)).cast<std::string>() +
           ", passed=" + std::to_string(report.passed()) +
           ", failed=" + std::to_string(report.failed()) + ")";
}

std::string describe_suite(const eval::Suite& suite) {
    return "Suite(name=" + py::repr(py::str(suite.name)).cast<std::string>() +
           ", jobs=" + std::to_string(suite.jobs.size()) + ")";
}

std::string describe_diff(const eval::ReportDiff& diff) {
    return "ReportDiff(jobs=" + std::to_string(diff.jobs.size()) +
           ", regressed=" + (diff.regressed() ? "True" : "False") + ")";
}

// A suite, or the path to a file containing one. Taking both is what lets the
// common case be one line and the two-step case still exist for a caller that
// wants to look at the suite before running it.
eval::Suite suite_from(const py::object& suite) {
    if (py::isinstance<eval::Suite>(suite)) {
        return suite.cast<eval::Suite>();
    }
    return eval::load_suite(py::cast<std::filesystem::path>(suite));
}

eval::SuiteReport run_suite_from(const py::object& suite, std::size_t jobs) {
    eval::Suite loaded = suite_from(suite);

    eval::SuiteReport report;
    {
        // Released for the length of the run, and unlike the run itself this
        // one is safe for a reason worth writing down: the workers replay
        // journals, and a replay never calls back into Python -- the provider
        // and the tools are rebuilt from the recording. Nothing on those
        // threads needs the GIL, so nothing on them may take it.
        py::gil_scoped_release release;
        report = jobs == 1 ? eval::run_suite(loaded) : eval::run_suite(loaded, jobs);
    }
    return report;
}

}  // namespace

void register_eval(py::module_& m) {
    py::module_ module = m.def_submodule(
        "eval", "The eval harness: replay a suite of recorded runs and report on them.");

    py::class_<eval::Suite>(module, "Suite", "A parsed suite file.")
        .def_readonly("name", &eval::Suite::name)
        .def_property_readonly("jobs", &jobs_of, "One dict per job, in suite order.")
        .def("__len__", [](const eval::Suite& suite) { return suite.jobs.size(); })
        .def("__repr__", &describe_suite);

    py::class_<eval::SuiteReport>(module, "SuiteReport", "What running a suite produced.")
        .def_readonly("suite", &eval::SuiteReport::suite)
        .def_property_readonly("passed", &eval::SuiteReport::passed)
        .def_property_readonly("failed", &eval::SuiteReport::failed)
        .def_property_readonly("total_prompt_tokens", &eval::SuiteReport::total_prompt_tokens)
        .def_property_readonly("total_completion_tokens",
                               &eval::SuiteReport::total_completion_tokens)
        .def_property_readonly("total_cost_usd", &eval::SuiteReport::total_cost_usd,
                               "Summed over the jobs whose model has a known price; pair it "
                               "with every_cost_known before quoting it.")
        .def_property_readonly("every_cost_known", &eval::SuiteReport::every_cost_known)
        .def_property_readonly("jobs", [](const eval::SuiteReport& report) {
            py::list jobs;
            for (const eval::JobResult& job : report.jobs) {
                jobs.append(to_python(job));
            }
            return jobs;
        })
        .def("latency_percentile_us", &eval::SuiteReport::latency_percentile_us,
             py::arg("percentile"),
             "The model-call latency at a percentile from 0 to 100, nearest rank -- so the "
             "answer is always a latency some call actually had.")
        .def("__repr__", &describe_report);

    py::class_<eval::ReportDiff>(module, "ReportDiff",
                                 "What changed between a baseline and a current report.")
        .def_property_readonly("jobs", &deltas_of)
        .def_readonly("notes", &eval::ReportDiff::notes)
        .def_property_readonly("regressed", &eval::ReportDiff::regressed,
                               "Whether anything got worse, a job disappearing included -- "
                               "deleting the job that fails is the cheapest way to make a "
                               "suite look better.")
        .def("__repr__", &describe_diff);

    module.def("load_suite", &eval::load_suite, py::arg("path"),
               R"(Parse a suite file. Raises ValueError naming the offending field.

Paths inside the suite are resolved relative to the suite file, so a suite and
its journals move together.)");

    module.def("run_suite", &run_suite_from, py::arg("suite"), py::arg("jobs") = 1,
               R"(Replay every job in a suite and report on it.

`suite` is either a parsed Suite or the path to one. Every job is answered from
its own journal: no network, no API key, no filesystem writes, and no tokens
spent. `jobs` is how many replay workers to use -- 1 runs on this thread and
never starts one, and 0 means one worker per core.

The report does not depend on `jobs`. Every number in it comes from a journal
rather than from the clock, and a batch that finished out of order is put back
into suite order before anything is written down.)");

    module.def("diff_reports", &eval::diff_reports, py::arg("baseline"), py::arg("current"),
               "Compare two reports. A job that disappeared counts as a regression.");

    module.def("write_report", &eval::write_report, py::arg("report"), py::arg("path"));
    module.def("read_report", &eval::read_report, py::arg("path"));
}

}  // namespace ash::python
