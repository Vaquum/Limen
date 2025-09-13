# Writing Docstrings

## Philosophy

### Documentation as Code Contract
Docstrings are not afterthoughts—they are the definitive specification of what your function does, what it expects, and what it returns. They serve as both user documentation and developer contract.

### Consistency Over Creativity
Uniform docstring structure removes cognitive overhead. When every function follows the same pattern, developers can scan parameters and returns instantly without parsing different formats.

### Exhaustive but Concise
Every parameter, every return value, every notable behavior must be documented. But verbose explanations are forbidden—clarity comes from precision, not verbosity.

### Source of Truth Synchronization
Docstrings and documentation must be identical. When code changes, both must be updated simultaneously. No exceptions.

### Machine-Readable Human-Friendly
Type hints are mandatory and must match docstring parameter types exactly. Descriptions should be clear enough for both humans and documentation generators.

### Semantic Coherence
Every word matters. Parameter names, descriptions, and terminology must be semantically coherent across the entire codebase. Similar concepts use identical language; different concepts use distinct language. No synonyms for the sake of variety.

## Workflow

It's highly advisable to work in an IDE such as Cursor, where the auto-complete already knows what we want to have in the docstring. One can then simply ask LLM to sync that with docs, and do few iterative rounds of consistency checks with the LLM. 

## Standard Format

Every function docstring in Loop follows this exact structure:

```python
def function_name(data: pl.DataFrame, param: int) -> pl.DataFrame:
    
    '''
    Compute [specific behavior] using [method/algorithm].
    
    Args:
        data (pl.DataFrame): [Klines/Trades] dataset with '[specific columns]' column[s]
        param (int): [Specific description of parameter purpose]
        
    Returns:
        pl.DataFrame: The input data with a new column '[exact_column_name]'
    '''
```

## Mandatory Rules

### 1. Verb Usage
- **ALWAYS** use "Compute" for calculations, transformations, and indicators
- **ONLY** use "Create" when genuinely creating new structures (rare)
- **NEVER** use "Calculate", "Generate", "Make", "Build", or other verbs

### 2. Data Parameter Description
- **ALWAYS** follow pattern: `"[Klines/Trades] dataset with '[specific columns]' column[s]"`
- **NEVER** use generic terms like "OHLC columns" or "price column"
- **Examples:**
  - ✅ `"Klines dataset with 'high', 'low', 'close' columns"`
  - ✅ `"Klines dataset with 'close' column"`
  - ✅ `"Trades dataset with 'price', 'quantity' columns"`
  - ❌ `"Klines dataset with OHLC columns"`
  - ❌ `"Dataset with price data"`

### 3. Parameter Descriptions
- **NEVER** include default values in descriptions
- **ALWAYS** be specific about parameter purpose
- **Examples:**
  - ✅ `"Number of periods for ATR calculation"`
  - ✅ `"Column name for price data"`
  - ❌ `"Number of periods for ATR calculation (default: 14)"`
  - ❌ `"Period parameter"`

### 4. Return Descriptions
- **ALWAYS** specify the exact column name(s) being added
- **ALWAYS** follow pattern: `"The input data with a new column '[exact_name]'"`
- **Examples:**
  - ✅ `"The input data with a new column 'atr'"`
  - ✅ `"The input data with three columns: 'macd', 'macd_signal', 'macd_hist'"`
  - ❌ `"DataFrame with ATR values"`
  - ❌ `"The input data with calculated values"`

### 5. NOTE Formatting
- **ALWAYS** use "NOTE:" (all caps with colon)
- **NEVER** use "Note:", "note:", or other variations
- **Examples:**
  - ✅ `"NOTE: Different from standard ATR which uses Wilder's smoothing."`
  - ❌ `"Note: Different from standard ATR..."`

## Structure Requirements

### Layout

#### Title
- No longer than a single sentence.
- Ends with a full stop
- Can have `NOTE` section below the heading

#### Args
- Definitions no longer than a single sentence
- No full stop at the end

#### Returns
- Definition no longer than a single sentence
- No full stop at the end

### Spacing (Authoritative Rules)
- There is exactly one empty line between the function declaration and the opening `'''`
- Inside the docstring:
  - One single-line title comes first
  - Then exactly one empty line
  - Then the `Args:` section (if present)
  - Then exactly one empty line
  - Then the `Returns:` section (if present)
  - There are no additional empty lines within sections themselves
- There is exactly one empty line after the closing `'''` before the code body begins

Decision flow for spacing:

1) Is there a docstring?
   - Yes → Ensure 1 blank line before opening `'''`
   - No  → Add a docstring

2) Inside the docstring, after the title line:
   - Always add 1 blank line

3) Next section present?
   - Args present → Write `Args:` block (no internal blank lines), then add 1 blank line
   - Returns only → Skip directly to `Returns:`

4) Returns present?
   - Yes → Write `Returns:` block (no internal blank lines)
   - No  → End docstring

5) After the closing `'''`:
   - Always add 1 blank line before code

### Sections Order
1. Main description (single line)
2. Optional NOTE section
3. Args section
4. Returns section
5. Optional additional NOTE sections

### Type Consistency
- Parameter types in docstring must match function signature exactly
- Use `pl.DataFrame`, `str`, `int`, `float`, `bool`, `list[str]`, etc.
- Optional parameters: `str, optional` or `str | None`

## Examples

### Simple Function
```python
def sma(data: pl.DataFrame, column: str, period: int) -> pl.DataFrame:
    
    '''
    Compute Simple Moving Average (SMA) indicator.
    
    Args:
        data (pl.DataFrame): Klines dataset with price column
        column (str): Column name to calculate SMA on
        period (int): Number of periods for SMA calculation
        
    Returns:
        pl.DataFrame: The input data with a new column '{column}_sma_{period}'
    '''
```

### Function with NOTE
```python
def rsi_sma(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    
    '''
    Compute RSI using Simple Moving Average smoothing (not Wilder's method).
    
    NOTE: Different from wilder_rsi which uses exponential smoothing.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods for RSI calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'rsi_sma'
    '''
```

### Function with Optional Parameters
```python
def lag_column(data: pl.DataFrame, 
               col: str,
               lag: int,
               alias: str = None) -> pl.DataFrame:
    
    '''
    Compute a lagged version of a column.

    Args:
        data (pl.DataFrame): Klines dataset with specified column
        col (str): The column name to lag
        lag (int): The number of periods to lag
        alias (str, optional): New column name. If None, uses alias f"lag_{lag}"

    Returns:
        pl.DataFrame: The input data with the lagged column appended
    '''
```

## Quality Checklist

Before submitting any function, verify:

- [ ] Uses "Compute" or "Create" (never "Calculate")
- [ ] Data parameter specifies exact column names
- [ ] No default values mentioned in parameter descriptions
- [ ] Return description specifies exact column name(s)
- [ ] "NOTE:" formatting (if applicable)
- [ ] Type hints match docstring types exactly
- [ ] Proper spacing (empty lines above/below)
- [ ] Documentation file updated with identical content

## Common Mistakes

### ❌ Wrong Verb
```python
'''
Calculate price change percentage...
'''
```

### ❌ Generic Data Description
```python
'''
Args:
    data (pl.DataFrame): Dataset with price columns
'''
```

### ❌ Default Values in Description
```python
'''
Args:
    period (int): Number of periods (default: 14)
'''
```

### ❌ Vague Return Description
```python
'''
Returns:
    pl.DataFrame: Data with new values
'''
```

### ❌ Wrong NOTE Format
```python
'''
Note: This is different from...
'''
```

## Documentation Synchronization

**CRITICAL:** Every docstring change requires corresponding documentation update:

1. Update function docstring
2. Update corresponding entry in `docs/Indicators.md` or `docs/Features.md`
3. Ensure content is **identical** between code and docs
4. Test imports after changes
5. Increment version in `setup.py`

The documentation is the user-facing contract. Inconsistency between code and docs is a critical bug.