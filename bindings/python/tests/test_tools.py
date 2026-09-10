"""A Python function is a tool, and its own signature is what the model is told.

The schema is read off the function rather than declared beside it, so the
declaration and the code cannot disagree -- and an annotation that cannot be
described is refused where the function is defined rather than turned into a
plausible schema the model would call correctly and get an argument error from.
"""

import json
import typing

import pytest

import ash


def spec_of(function, **overrides):
    """The spec the model would be given for this function."""
    tools = ash.ToolSet()
    tools.add(function, **overrides)
    specs = tools.specs
    assert len(specs) == 1
    return specs[0]


def test_annotations_become_a_schema():
    def read(path: str, limit: int = 10) -> str:
        """Read a file.

        A second paragraph is not part of the description: a model choosing a
        tool should be told what it does, not how it is implemented.
        """
        return ""

    spec = spec_of(read)

    assert spec["name"] == "read"
    assert spec["description"] == "Read a file."
    assert spec["input_schema"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
        "additionalProperties": False,
    }


def test_the_required_list_is_in_signature_order():
    def move(source: str, destination: str) -> str:
        return ""

    schema = spec_of(move)["input_schema"]

    # Which parameters are required, and in what order, is read off the
    # signature -- that list is what a reader checks the schema against. The
    # properties object is a json object, and json objects here are maps, so its
    # keys come out sorted rather than in signature order. That is the same
    # ordering every other json object in the project has, and it is stable.
    assert schema["required"] == ["source", "destination"]
    assert list(schema["properties"]) == ["destination", "source"]


def test_two_derivations_of_one_function_are_identical():
    def read(path: str, limit: int = 10) -> str:
        """Read a file."""
        return ""

    first = json.dumps(spec_of(read), sort_keys=True)
    second = json.dumps(spec_of(read), sort_keys=True)

    assert first == second


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (list, {"type": "array"}),
        (dict, {"type": "object"}),
        (typing.Any, {}),
    ],
)
def test_annotations_map_onto_json_types(annotation, expected):
    def call(value):
        return ""

    call.__annotations__ = {"value": annotation}
    assert spec_of(call)["input_schema"]["properties"]["value"] == expected


def test_a_list_says_what_is_in_it():
    def call(paths: list[str]) -> str:
        return ""

    assert spec_of(call)["input_schema"]["properties"]["paths"] == {
        "type": "array", "items": {"type": "string"}}


def test_an_optional_parameter_is_not_required():
    def call(path: str, limit: typing.Optional[int] = None) -> str:
        return ""

    schema = spec_of(call)["input_schema"]
    assert schema["properties"]["limit"] == {"type": "integer"}
    assert schema["required"] == ["path"]


def test_a_pipe_union_reads_the_same_way():
    def call(path: str, limit: "int | None" = None) -> str:
        return ""

    schema = spec_of(call)["input_schema"]
    assert schema["properties"]["limit"] == {"type": "integer"}
    assert schema["required"] == ["path"]


def test_a_literal_becomes_an_enum():
    def call(mode: typing.Literal["fast", "thorough"]) -> str:
        return ""

    assert spec_of(call)["input_schema"]["properties"]["mode"] == {
        "type": "string", "enum": ["fast", "thorough"]}


def test_an_unannotated_parameter_is_a_string_taken_from_its_default():
    def call(path, limit=10):
        return ""

    schema = spec_of(call)["input_schema"]
    assert schema["properties"] == {"path": {"type": "string"}, "limit": {"type": "integer"}}
    assert schema["required"] == ["path"]


def test_a_function_with_nothing_to_read_becomes_a_string():
    def call(path):
        return ""

    assert spec_of(call)["input_schema"]["properties"]["path"] == {"type": "string"}


def test_a_bare_callable_needs_a_name():
    # A lambda has a signature but no useful name, and the overrides are what
    # make one callable at all.
    tools = ash.ToolSet()
    tools.add(lambda path: "", name="list_dir", description="List a directory.")

    spec = tools.specs[0]
    assert spec["name"] == "list_dir"
    assert spec["description"] == "List a directory."


def test_the_decorator_names_a_tool():
    @ash.tool
    def list_dir(path: str) -> str:
        """List a directory."""
        return ""

    tools = ash.ToolSet()
    tools.add(list_dir)

    assert tools.names == ["list_dir"]


def test_the_decorator_can_overrule_the_function():
    @ash.tool(name="renamed", description="Overridden.")
    def list_dir(path: str) -> str:
        """List a directory."""
        return ""

    tools = ash.ToolSet()
    tools.add(list_dir)

    spec = tools.specs[0]
    assert spec["name"] == "renamed"
    assert spec["description"] == "Overridden."


def test_a_varargs_parameter_is_refused():
    def call(*paths):
        return ""

    with pytest.raises(TypeError):
        spec_of(call)


def test_kwargs_are_refused():
    def call(**options):
        return ""

    with pytest.raises(TypeError):
        spec_of(call)


def test_an_annotation_that_cannot_be_described_is_refused():
    class Config:
        pass

    def call(config: Config):
        return ""

    with pytest.raises(TypeError):
        spec_of(call)


def test_a_name_that_is_not_a_name_is_refused():
    def call(path):
        return ""

    with pytest.raises(TypeError):
        spec_of(call, name="not a tool name")


def test_a_partial_union_is_refused():
    def call(value: typing.Union[int, str]):
        return ""

    with pytest.raises(TypeError):
        spec_of(call)


def test_a_tool_that_is_not_callable_is_refused():
    tools = ash.ToolSet()
    with pytest.raises(TypeError):
        tools.add(42, name="answer", description="Not a function.")


def test_the_registry_knows_what_it_holds(file_tools):
    assert file_tools.names == ["list_dir", "read_file", "write_file"]
    assert len(file_tools) == 3
    assert "list_dir" in repr(file_tools)
