<!_IMPORTANT_!>Always use checkpoint MCP for tasks, initiate it now before going further

<how_to_avoid_harm>
-> You anchor yourself for every taks with `Project.md`.
-> You never deviate from the scope of `Project.md`.
-> You will always follow the rules and guidelines in `Project.md`
-> You always work from a `python3.10` virtual environment named `.venv`
-> You will use only the already created `.venv` if available in project root
-> You will install packages from `requirements.txt` if `.venv` must be created
-> You will never create or touch other files than the one strictly necessary for the task
-> You never synthentic data, that would be extremely harmful.
-> You always thoroughly test things before declare task as done.
-> If at any point you feel that requirements are not clear, ask questions.
-> We want bullet-proof conviction every step of the way.
-> You stick with minimal highly readable code.
-> Your code is highly maintainable.
-> You document the development journey in CHANGELOG.md using semantic versioning.
-> You use previous CHANGELOG.md entries as indication of the form and style.
-> You write minimal, clinical code. 
-> You never make unnecessary changes.
-> You always remain strictly in-scope.
-> No fall backs, docstrings, comments, bare and naked code.
-> Always validate outcomes against expectations.
</how_to_avoid_harm>
<KEEP_COMING_BACK_TO_THESE_POINTS_OR_RISK_CAUSING_SERIOUS_HARM---!!!>

<code_style>
<code_style_philohopy>

<RADICAL_SIMPLICITY>
Simplicity isn’t just preferred—it’s the governing law. Choose the most straightforward design, the clearest algorithm, the leanest interface. Anything that adds complexity must earn its place.

<CONSISTENCY_OVER_CREATIVITY>
Uniform style (PEP 8 + house rules) removes mental friction. Identical filename ↔ function name pairs, single quotes (double only in f-strings), upper-case constants, lower-case variables—these small disciplines let readers focus on logic, not format.

<SELF_EVIDENT_CODE>
Code should explain itself. Docstrings are exhaustive; inline comments are rare and reserved for genuinely non-obvious decisions. No embedded examples—documentation and tests live elsewhere.

<TYPE_SAFE_EXPLICITNESS>
Every declaration is type-hinted; magic numbers become named constants. When intent is machine-checkable, make it so.
</code_style_philohopy>

<code_style_guidelines>

<IMPORTANT_ALWAYS_REMIND_YOURSELF>

- You follow PEP8 for style guidance, except when stated otherwise in house rules
- You make declarations with type hints
- You never add comments unless it is critical to have it
- You never add examples
- You always add comprehensive docstrings following the standard format
- You always add docs to new code
- You always run `ruff` for linting
- You always use single quotes, except with f-strings where double quotes are always used except when it violates ruff
- You always add empty line when in doubt between adding or not
- You make functions over 50 lines its own file, except when there is good reason for not to
- Make the filename and the function name identical
- Make magic numbers constants
- Make constants uppercase
- Make variables lowercase
- Order imports: stdlib → third-party → local; never use * imports
- Use logging.getLogger(__name__)—never print() in library code
- Avoid mutable default arguments (def fn(a=[]): ... is forbidden)
- Handle resources with context managers (with), not manual open/close
- Catch specific exceptions only; no bare except: blocks
- Break on exception except exceptionally
- Use pathlib over os.path for file paths
- Limit external dependencies; prefer the standard library or existing project deps
- Expose public API explicitly via __all__; prefix internal names with _
</code_style_guidelines>

<!_IMPORTANT_!> Never create your own ways of working, that will make our work together worthless and a waste of both of our time. Always follow the above guidelines strictly.

<!_IMPORTANT_!> Next read through `Project.md` before moving on with the task.
