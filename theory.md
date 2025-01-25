# Compiler Theory Principles in Clox Implementation

## Lexical Analysis (Scanning)
The first major compiler concept in Clox is lexical analysis. This is handled by the scanner, which breaks source code into meaningful tokens. The scanner applies several key theoretical concepts:

### 1. Regular Languages and Finite Automata
In `scanner.c`, the `scanToken()` function essentially implements a finite state machine. For example, when scanning numbers:

```c
static Token number() {
  while (isDigit(peek())) advance();

  // Look for a fractional part.
  if (peek() == '.' && isDigit(peekNext())) {
    // Consume the "."
    advance();

    while (isDigit(peek())) advance();
  }
  // ...
}
```

This code implements a deterministic finite automaton (DFA) that recognizes numeric literals. The states are:
- Initial state (seeing first digit)
- Reading integer part
- Seeing decimal point
- Reading fractional part

## Syntactic Analysis (Parsing)
Clox uses recursive descent parsing, which is based on several theoretical concepts:

### 1. Context-Free Grammars (CFG)
The language's syntax is implicitly defined by the parser's structure. For example, the expression grammar is implemented through parsing rules:

```c
ParseRule rules[] = {
  [TOKEN_PLUS]          = {NULL,     binary, PREC_TERM},
  [TOKEN_MINUS]         = {unary,    binary, PREC_TERM},
  [TOKEN_STAR]          = {NULL,     binary, PREC_FACTOR},
  // ...
};
```

This table represents the CFG productions for expressions. Each entry tells us:
- What can appear on the left of the operator (prefix rule)
- What can appear on the right (infix rule)
- The operator's precedence level

### 2. Operator Precedence and Associativity
Clox implements the theoretical concept of precedence climbing through its `parsePrecedence` function:

```c
static void parsePrecedence(Precedence precedence) {
  advance();
  ParseFn prefixRule = getRule(parser.previous.type)->prefix;
  if (prefixRule == NULL) {
    error("Expect expression.");
    return;
  }

  bool canAssign = precedence <= PREC_ASSIGNMENT;
  prefixRule(canAssign);
  // ...
}
```

This implements a precedence climbing algorithm, which is a way to parse expressions that:
- Respects operator precedence
- Handles left and right associativity correctly
- Is more efficient than a pure grammar-based approach

## Semantic Analysis
Clox performs several types of semantic analysis during compilation:

### 1. Scope Analysis
The compiler maintains a stack of local variables and their scope depths:

```c
typedef struct {
  Local locals[UINT8_COUNT];
  int localCount;
  int scopeDepth;
  // ...
} Compiler;
```

This implements the theoretical concept of lexical scoping, where:
- Variables are only visible in their declaring block and nested blocks
- Inner scopes can shadow outer variables
- The compiler can determine variable visibility at compile time

### 2. Type Checking (Limited)
While Clox is dynamically typed, it still performs some type-related checks at runtime through its value system:

```c
typedef struct {
  ValueType type;
  union {
    bool boolean;
    double number;
    Obj* obj;
  } as;
} Value;
```

## Code Generation
Clox uses several code generation techniques from compiler theory:

### 1. Stack-Based Virtual Machine
The compiler generates bytecode for a stack-based VM, which implements the theoretical concept of abstract machines:

```c
void binary(bool canAssign) {
  TokenType operatorType = parser.previous.type;
  ParseRule* rule = getRule(operatorType);
  parsePrecedence((Precedence)(rule->precedence + 1));

  switch (operatorType) {
    case TOKEN_PLUS:          emitByte(OP_ADD); break;
    case TOKEN_MINUS:         emitByte(OP_SUBTRACT); break;
    // ...
  }
}
```

### 2. Static Single Assignment (SSA)
While Clox doesn't explicitly use SSA form, its stack-based nature achieves similar benefits:
- Each stack slot holds exactly one value at a time
- Operations always work with the current top of stack
- No need for explicit variable renaming

## Optimization
Clox implements some basic optimizations:

### 1. Constant Folding
When defining constants:
```c
static void number(bool canAssign) {
  double value = strtod(parser.previous.start, NULL);
  emitConstant(NUMBER_VAL(value));
}
```
