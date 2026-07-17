import sys

import atheris

with atheris.instrument_imports():
    from limen.yaml import parse
    from limen.yaml import validate

MAX_TEXT_CHARS = 4096


def fuzz_yaml_manifest(data: bytes) -> None:

    '''
    Drive the manifest parse and validate chain with fuzzed YAML text.

    Both surfaces promise error collection without raising: parse
    returns a dict and an error list for arbitrary text, and validate
    returns a ValidationResult for arbitrary parsed mappings, so any
    uncaught exception under this harness is a genuine finding.

    Args:
        data (bytes): Fuzzer-generated input buffer

    Returns:
        None: The function returns nothing

    '''

    provider = atheris.FuzzedDataProvider(data)
    text = provider.ConsumeUnicodeNoSurrogates(MAX_TEXT_CHARS)
    parsed, errors = parse(text)

    if not errors:
        validate(parsed)


def main() -> None:

    '''
    Run the atheris fuzzing loop with CLI-provided libFuzzer flags.

    Returns:
        None: The function returns nothing

    '''

    atheris.Setup(sys.argv, fuzz_yaml_manifest)
    atheris.Fuzz()


if __name__ == '__main__':
    main()
