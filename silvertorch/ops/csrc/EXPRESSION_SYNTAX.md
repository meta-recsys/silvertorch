# Expression Query Syntax

A text-based query language for silvertorch bloom index filtering.
Produces the same query plan as the thrift-based `parse_filter_query_batch`.

## Feature Terms (Leaf Nodes)

A feature term matches a single feature id/value pair:

```
id:value            e.g.  42:100
id:value:weight     e.g.  42:100:0.5
```

- `id` — integer, the feature id
- `value` — integer (may be negative), the feature value
- `weight` — optional float, the feature weight

## Logical Operators

| Operator | Keyword | Symbol | Description |
|----------|---------|--------|-------------|
| NOT      | `NOT`   | `!`    | Negates the following expression |
| AND      | `AND`   | `&`    | All operands must match |
| OR       | `OR`    | `\|`   | At least one operand must match |

### Precedence (high to low)

1. `NOT` / `!`
2. `AND` / `&`
3. `OR` / `|`

Use parentheses `()` to override precedence.

## Examples

```
# Single feature match
42:100

# Feature with weight
42:100:0.5

# AND — both features must match
1:100 AND 2:200
1:100 & 2:200

# OR — either feature matches
1:100 OR 2:200
1:100 | 2:200

# NOT — exclude a feature
NOT 3:300
!3:300

# Nested with precedence
(1:100 AND 2:200) OR NOT 3:300

# Complex
(1:100 & 2:200) | !(3:300 & 4:400:0.8)

# Precedence: NOT > AND > OR, so these are equivalent:
1:10 OR 2:20 AND NOT 3:30
1:10 OR (2:20 AND (NOT 3:30))
```

## Grammar (EBNF)

```
expression  = or_expr ;
or_expr     = and_expr { ("OR" | "|") and_expr } ;
and_expr    = unary_expr { ("AND" | "&") unary_expr } ;
unary_expr  = ("NOT" | "!") unary_expr | primary ;
primary     = "(" expression ")" | feature_term ;
feature_term = NUMBER ":" NUMBER [ ":" NUMBER ] ;
NUMBER      = ["-"] DIGIT+ ["." DIGIT+] ;
```

## Usage (Python)

```python
import torch

# Single query
max_stack, tensors = torch.ops.st.parse_expression_query_batch(
    ["1:100 AND 2:200"],       # expressions (batch)
    silvertorch_ks,             # silvertorch k values
    bloom_hash_k=7,             # bloom hash k
)

# Batch of queries
max_stack, tensors = torch.ops.st.parse_expression_query_batch(
    ["1:100 AND 2:200", "NOT 3:300", "4:400 | 5:500"],
    silvertorch_ks,
    bloom_hash_k=7,
)
```
