# Compiler Theory Principles in Clox Implementation

## CLOX DEPENDENT THEORY

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

# CLOX INDEPENDENT THEORY

# Lexical Analysis Theory and Techniques

## 1. Regular Languages

### a) Regular Expressions
Regular expressions form the foundation of lexical analysis. Key example in Clox:

```c
// This code implicitly implements the regular expression [0-9]+(\.[0-9]+)?
static Token number() {
  while (isDigit(peek())) advance();
  if (peek() == '.' && isDigit(peekNext())) {
    advance();
    while (isDigit(peek())) advance();
  }
}
```

### b) Finite Automata
Two main types:

- **Deterministic Finite Automata (DFA)** - Used in Clox
  - Single transition per input
  - Faster execution
  - Larger state tables

- **Nondeterministic Finite Automata (NFA)**
  - Multiple transitions per input
  - More compact representation
  - Requires backtracking

## 2. Scanner Implementation Techniques

### a) Hand-Written Scanners (Clox Approach)
```c
static TokenType identifierType() {
  switch (scanner.start[0]) {
    case 'a': return checkKeyword(1, 2, "nd", TOKEN_AND);
    case 'c': return checkKeyword(1, 4, "lass", TOKEN_CLASS);
    // ...
  }
}
```

Advantages:
- Direct error handling control
- Higher efficiency
- Easier debugging

### b) Table-Driven Scanners
Example in Flex:
```flex
%%
[0-9]+          { return NUMBER; }
[a-zA-Z_][a-zA-Z0-9_]* { return IDENTIFIER; }
"if"            { return IF; }
%%
```

Benefits:
- Better maintainability
- Easy language modification
- Automatic error handling

## 3. Advanced Scanning Techniques

### a) Lookahead Buffering
```c
class Scanner {
    private char[] buffer;
    private int current;
    private int lookAhead;
    
    char peek(int distance) {
        return buffer[(current + distance) % BUFFER_SIZE];
    }
}
```

Used for:
- Error recovery
- Efficient token recognition
- Complex token handling

### b) Lazy Lexical Analysis
```c
class LazyScanner {
    private String source;
    private int position;
    
    Token nextToken() {
        if (tokenCache.isEmpty()) {
            scanNextToken();
        }
        return tokenCache.pop();
    }
}
```

Benefits:
- Improved efficiency
- Incremental compilation support
- IDE integration

## 4. Error Handling Strategies

### a) Basic Error Recovery (Clox)
```c
static void skipWhitespace() {
  for (;;) {
    char c = peek();
    switch (c) {
      case ' ':
      case '\r':
      case '\t':
      case '\n':
        advance();
        break;
      default:
        return;
    }
  }
}
```

### b) Advanced Error Recovery
```c
class SophisticatedScanner {
    void recover() {
        while (!isAtSynchronizationPoint()) {
            advance();
        }
        resynchronizeState();
        reportError();
    }
}
```

Features:
- Error correction
- Multiple error reporting
- Context-aware recovery

## 5. Character Stream Management

### a) Direct Reading (Clox)
```c
static char advance() {
  scanner.current++;
  return scanner.current[-1];
}
```

### b) Buffered Reading
```c
class BufferedScanner {
    private char[] buffer;
    private int bufferSize;
    
    char getNext() {
        if (bufferPos >= bufferSize) {
            refillBuffer();
        }
        return buffer[bufferPos++];
    }
}
```

Benefits:
- Better large file performance
- File encoding support
- Improved line ending handling

---------------------------------------------------------------------------------------------


## INDIVIDUAL TOPICS

# Finite Automata in Lexical Analysis

## Finite Automata Fundamentals

### 1. Deterministic Finite Automata (DFA)
Clox's implementation of DFA for number scanning:

```c
static Token number() {
  while (isDigit(peek())) advance();  // State 1: Integer part
  
  if (peek() == '.' && isDigit(peekNext())) {
    advance();  // State 2: Decimal point
    while (isDigit(peek())) advance();  // State 3: Fractional part
  }
  
  return makeToken(TOKEN_NUMBER);
}
```

States:
- Initial state: First digit
- State 1: Integer digits
- State 2: Decimal point
- State 3: Fractional digits
- Final state: Complete number

### 2. Nondeterministic Finite Automata (NFA)
Multiple possible next states for the same input.

## Modern Language Implementations

### 1. Rust (logos)
```rust
#[derive(Logos, Debug, PartialEq)]
enum Token {
    #[regex(r"[0-9]+(\.[0-9]+)?")]
    Number,
    
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier,
}
```

### 2. Go
```go
func (l *lexer) scanNumber() token {
    // Start with integer part
    l.acceptRun("0123456789")
    
    // Optional decimal part
    if l.accept(".") {
        l.acceptRun("0123456789")
    }
    
    // Optional exponent
    if l.accept("eE") {
        l.accept("+-")
        l.acceptRun("0123456789")
    }
    return l.emit(tokenNumber)
}
```

### 3. Python (3.9+)
```python
# In Python's Grammar definition
number: NUMBER
NUMBER: "-"? DIGIT+ ("." DIGIT+)? ("e" [+-]? DIGIT+)?
DIGIT: "0"..."9"
```

## Implementation Comparisons

### Clox's Simple DFA
```c
static Token string() {
  while (peek() != '"' && !isAtEnd()) {
    if (peek() == '\n') scanner.line++;
    advance();
  }
  
  if (isAtEnd()) return errorToken("Unterminated string.");
  
  // The closing quote.
  advance();
  return makeToken(TOKEN_STRING);
}
```

### Modern Production Compiler (V8)
```cpp
Token Scanner::ScanString() {
  StartString();
  while (true) {
    if (c0_ == '"') break;
    if (c0_ == '\\') {
      AdvanceUntil('\n');
      continue;
    }
    if (IsLineTerminator(c0_)) {
      HandleMultilineString();
    }
    AddToString(c0_);
    Advance();
  }
  return FinalizeString();
}
```

### Key Implementation Differences
1. Error Recovery
2. Unicode Support
3. Performance Optimizations
4. State Management




# PARSING THEORY AND IMPLEMENTATION

### Context-Free Grammars (CFGs)
This is the theoretical foundation for describing programming language syntax. Let me explain with a concrete example:

In Clox, we implicitly define our grammar through code, but it could be expressed formally like this:
```
expression → literal | unary | binary | grouping
literal    → NUMBER | STRING | "true" | "false" | "nil"
unary      → ("-" | "!") expression
binary     → expression operator expression
operator   → "+" | "-" | "*" | "/"
```

Each rule shows how language constructs can be built from simpler parts. Clox implements this through its ParseRule table:

```c
ParseRule rules[] = {
  [TOKEN_LEFT_PAREN]    = {grouping, NULL,   PREC_NONE},
  [TOKEN_RIGHT_PAREN]   = {NULL,     NULL,   PREC_NONE},
  [TOKEN_LEFT_BRACE]    = {NULL,     NULL,   PREC_NONE},
  [TOKEN_RIGHT_BRACE]   = {NULL,     NULL,   PREC_NONE},
  [TOKEN_COMMA]         = {NULL,     NULL,   PREC_NONE},
  [TOKEN_DOT]          = {NULL,     NULL,   PREC_NONE},
  [TOKEN_MINUS]         = {unary,    binary, PREC_TERM},
  [TOKEN_PLUS]          = {NULL,     binary, PREC_TERM},
  [TOKEN_SEMICOLON]     = {NULL,     NULL,   PREC_NONE},
  [TOKEN_SLASH]         = {NULL,     binary, PREC_FACTOR},
  [TOKEN_STAR]          = {NULL,     binary, PREC_FACTOR}
};
```

Modern languages often use more sophisticated grammar specifications. For example, TypeScript uses a formal grammar specification:

```typescript
// TypeScript grammar excerpt
InterfaceDeclaration:
    interface Identifier TypeParameters? InterfaceExtendsClause? ObjectType

TypeParameters:
    < TypeParameterList >

TypeParameterList:
    TypeParameter
    TypeParameterList , TypeParameter
```

## Parsing Strategies

Let's compare different parsing approaches:

a) Recursive Descent (Used in Clox):
```c
static void expression() {
  parsePrecedence(PREC_ASSIGNMENT);
}

static void binary(bool canAssign) {
  TokenType operatorType = parser.previous.type;
  ParseRule* rule = getRule(operatorType);
  parsePrecedence((Precedence)(rule->precedence + 1));
}
```

This approach is:
- Intuitive to write and debug
- Directly mirrors the grammar
- Good for handling operator precedence

b) LR Parsing (Used in many production compilers):
```cpp
// Example from GCC's parser
void Parser::ParseStatement() {
    switch (current_token) {
        case T_IF:
            ParseIfStatement();
            break;
        case T_WHILE:
            ParseWhileStatement();
            break;
        // ...
    }
}
```

Benefits:
- Can handle more complex grammars
- Often more efficient
- Better error recovery

3. Abstract Syntax Tree (AST) Construction

Clox doesn't build an explicit AST, instead generating bytecode directly. However, most modern compilers do. Let's compare:

Clox's Direct Bytecode Generation:
```c
static void number(bool canAssign) {
  double value = strtod(parser.previous.start, NULL);
  emitConstant(NUMBER_VAL(value));
}
```

Modern Compiler's AST Approach (e.g., Rust's rustc):
```rust
enum Expr {
    Binary {
        left: Box<Expr>,
        operator: Token,
        right: Box<Expr>
    },
    Literal(Value),
    Unary {
        operator: Token,
        operand: Box<Expr>
    }
}
```

4. Error Recovery Strategies

Clox uses a simple panic mode:
```c
static void synchronize() {
  parser.panicMode = false;
  
  while (parser.current.type != TOKEN_EOF) {
    if (parser.previous.type == TOKEN_SEMICOLON) return;
    
    switch (parser.current.type) {
      case TOKEN_CLASS:
      case TOKEN_FUN:
      case TOKEN_VAR:
      case TOKEN_FOR:
      case TOKEN_IF:
      case TOKEN_WHILE:
      case TOKEN_PRINT:
      case TOKEN_RETURN:
        return;
      default:
        ; // Do nothing.
    }
    advance();
  }
}
```

Modern compilers use more sophisticated error recovery:
```typescript
// TypeScript's error recovery
class Parser {
    private skipToNextStatement() {
        while (!this.isAtStatementBoundary()) {
            this.advance();
            
            // Try to recover at statement boundaries
            if (this.isRecoveryToken(this.current())) {
                break;
            }
        }
        
        // Attempt to reconstruct the AST
        this.repairAst();
    }
}
```

5. Precedence and Associativity

Clox handles precedence through its precedence climbing algorithm:
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

Modern compilers might use precedence tables:
```cpp
enum Precedence {
    PREC_LOWEST,
    PREC_ASSIGN,    // =
    PREC_OR,        // ||
    PREC_AND,       // &&
    PREC_EQUALITY,  // == !=
    PREC_COMPARE,   // < > <= >=
    PREC_TERM,      // + -
    PREC_FACTOR,    // * /
    PREC_UNARY,     // ! -
    PREC_CALL,      // . () []
    PREC_PRIMARY
};
```


# PART 1: FUNDAMENTAL PARSING THEORY

1. Grammar Theory and Language Classification

The Chomsky Hierarchy defines four types of grammars, each with increasing power and complexity:
- Type 3: Regular Grammars (handled by finite automata)
- Type 2: Context-Free Grammars (most programming language syntax)
- Type 1: Context-Sensitive Grammars (some programming language semantics)
- Type 0: Unrestricted Grammars (Turing complete)

Programming languages primarily use Context-Free Grammars (CFGs). Let's see how this works in practice. Consider this simple expression:

```c
a = b + c * d
```

The CFG for this might look like:
```
expr → assign | term
assign → IDENTIFIER "=" expr
term → factor (("+"|"-") factor)*
factor → primary (("*"|"/") primary)*
primary → IDENTIFIER | NUMBER | "(" expr ")"
```

2. Grammar Ambiguity and Resolution

Consider this ambiguous grammar:
```
expr → expr "+" expr | expr "-" expr | NUMBER
```

This grammar is ambiguous because "1 + 2 + 3" could be parsed as either:
- (1 + 2) + 3
- 1 + (2 + 3)

Modern languages resolve this through precedence rules. For example, in Clox:

```c
typedef enum {
  PREC_NONE,
  PREC_ASSIGNMENT,  // =
  PREC_OR,         // or
  PREC_AND,        // and
  PREC_EQUALITY,   // == !=
  PREC_COMPARISON, // < > <= >=
  PREC_TERM,       // + -
  PREC_FACTOR,     // * /
  PREC_UNARY,      // ! -
  PREC_CALL,       // . ()
  PREC_PRIMARY
} Precedence;
```

PART 2: PARSING STRATEGIES IN DEPTH

1. Recursive Descent Parsing (Used in Clox)

Let's see how Clox parses an if statement:

```c
static void ifStatement() {
  consume(TOKEN_LEFT_PAREN, "Expect '(' after 'if'.");
  expression();
  consume(TOKEN_RIGHT_PAREN, "Expect ')' after condition.");

  int thenJump = emitJump(OP_JUMP_IF_FALSE);
  emitByte(OP_POP);
  statement();

  int elseJump = emitJump(OP_JUMP);
  patchJump(thenJump);
  emitByte(OP_POP);

  if (match(TOKEN_ELSE)) statement();
  patchJump(elseJump);
}
```

2. LR Parsing (Used in many system programming languages)

Consider how GCC might parse the same if statement:

```cpp
class Parser {
    Node* parseIfStatement() {
        Token token = expect(T_IF);
        expect(T_LPAREN);
        Node* condition = parseExpression();
        expect(T_RPAREN);
        
        Node* thenBranch = parseStatement();
        Node* elseBranch = nullptr;
        
        if (match(T_ELSE)) {
            elseBranch = parseStatement();
        }
        
        return new IfNode(token, condition, thenBranch, elseBranch);
    }
};
```

PART 3: INTERACTION WITH OTHER COMPILATION PHASES

1. Parsing to Semantic Analysis

The parser builds structures that the semantic analyzer uses. Let's see how this works for variable declarations:

```c
// Parser generates AST node
Node* parseVarDecl() {
    Token name = expect(T_IDENTIFIER);
    Node* initializer = nullptr;
    
    if (match(T_EQUALS)) {
        initializer = parseExpression();
    }
    
    expect(T_SEMICOLON);
    return new VarDeclNode(name, initializer);
}

// Semantic analyzer uses AST
void analyzeVarDecl(VarDeclNode* node) {
    // Check if variable is already declared
    if (symbolTable.contains(node->name)) {
        error("Variable already declared");
    }
    
    // Analyze initializer
    if (node->initializer) {
        Type* initType = analyzeExpr(node->initializer);
        checkTypeCompatibility(node->type, initType);
    }
    
    // Add to symbol table
    symbolTable.add(node->name, node->type);
}
```

2. Parser Integration with Type Checking

Modern languages like Rust combine parsing with immediate type checking:

```rust
impl<'a> Parser<'a> {
    fn parse_let_statement(&mut self) -> Result<Stmt, Error> {
        let name = self.expect_identifier()?;
        let type_annotation = if self.match_token(TokenType::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.expect_token(TokenType::Equals)?;
        let initializer = self.parse_expression()?;
        
        // Immediate type checking
        let inferred_type = self.type_checker.check_expression(&initializer)?;
        if let Some(annotated_type) = type_annotation {
            self.type_checker.unify(annotated_type, inferred_type)?;
        }
        
        Ok(Stmt::Let { name, type_annotation, initializer })
    }
}
```


# PARSING CHALLENGES IN DIFFERENT LANGUAGES

1. JavaScript's Automatic Semicolon Insertion (ASI)
This is one of the most interesting parsing challenges. JavaScript's parser needs to decide when to automatically insert semicolons. Consider this code:

```javascript
return
{
    key: value
}
```

The parser needs to handle this ambiguity. Here's how V8 (Chrome's JavaScript engine) approaches it:

```cpp
class Parser {
    bool needsAutomaticSemicolon() {
        // Check for line terminator
        if (scanner()->HasLineTerminatorBeforeNext()) return true;
        
        // Check for end of input
        if (scanner()->current_token() == Token::EOS) return true;
        
        // Check for closing tokens
        if (scanner()->current_token() == Token::RBRACE) return true;
        
        return false;
    }
};
```

2. Python's Significant Whitespace
Python's parser must track indentation levels as part of its syntax. This creates unique challenges. Here's how Python's parser handles it:

```python
class Parser:
    def handle_indent(self):
        current_indent = self.count_indent()
        if current_indent > self.indent_stack[-1]:
            # Starting new block
            self.indent_stack.append(current_indent)
            return INDENT
        elif current_indent < self.indent_stack[-1]:
            # Ending one or more blocks
            while current_indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                yield DEDENT
```

# ADVANCED PARSING TECHNIQUES

1. Generalized LR (GLR) Parsing
GLR parsing can handle ambiguous grammars by exploring multiple parsing paths simultaneously. Here's a conceptual implementation:

```cpp
class GLRParser {
    struct ParseState {
        Stack stack;
        AST* partial_tree;
    };
    
    vector<ParseState> active_states;
    
    void parse(Token token) {
        vector<ParseState> new_states;
        
        for (ParseState& state : active_states) {
            // Try all possible actions for this state
            auto actions = getActions(state.stack.top(), token);
            
            for (Action action : actions) {
                ParseState new_state = state;
                if (apply_action(new_state, action)) {
                    new_states.push_back(new_state);
                }
            }
        }
        
        active_states = move(new_states);
    }
};
```

2. Parsing Expression Grammars (PEG)
PEG parsers are becoming increasingly popular due to their unambiguous nature. Here's how they handle choice differently from CFGs:

```rust
// PEG-style parser combinator
fn parse_expression<'a>(&mut self, input: &'a str) -> Result<(Expr, &'a str)> {
    // Try alternatives in order, first success wins
    self.parse_assignment(input)
        .or_else(|_| self.parse_binary(input))
        .or_else(|_| self.parse_unary(input))
        .or_else(|_| self.parse_primary(input))
}
```

# INTERACTION WITH OPTIMIZATION PHASES

The parser's output significantly affects optimization opportunities. Let's explore this relationship:

1. Expression Tree Optimization
The parser can build ASTs that facilitate constant folding:

```cpp
class ExpressionParser {
    Node* parseBinaryExpr() {
        Node* left = parsePrimary();
        Token op = current_token;
        Node* right = parsePrimary();
        
        // Direct optimization during parsing
        if (left->isConstant() && right->isConstant()) {
            return foldConstants(left, op, right);
        }
        
        return new BinaryNode(left, op, right);
    }
    
    Node* foldConstants(Node* left, Token op, Node* right) {
        switch (op.type) {
            case PLUS:
                return new ConstantNode(left->value + right->value);
            case MULTIPLY:
                return new ConstantNode(left->value * right->value);
            // ...
        }
    }
};
```

2. Structure Analysis for Loop Optimization
The parser can identify loop structures that are candidates for optimization:

```cpp
class Parser {
    Node* parseForLoop() {
        ForLoop* loop = new ForLoop();
        
        // Parse loop structure
        loop->init = parseInitializer();
        loop->condition = parseCondition();
        loop->increment = parseIncrement();
        loop->body = parseBody();
        
        // Add optimization hints
        loop->analysis.is_countable = isCountableLoop(loop);
        loop->analysis.invariant_candidates = findInvariants(loop);
        
        return loop;
    }
    
    bool isCountableLoop(ForLoop* loop) {
        // Check if loop has a known number of iterations
        return loop->init->isSimpleInitializer() &&
               loop->condition->isSimpleComparison() &&
               loop->increment->isSimpleIncrement();
    }
};
```

3. Inlining Decisions
The parser can gather information that helps the optimizer make inlining decisions:

```cpp
class FunctionParser {
    FunctionNode* parseFunction() {
        FunctionNode* func = new FunctionNode();
        
        // Parse function body
        func->body = parseFunctionBody();
        
        // Gather optimization information
        func->analysis.size = measureFunctionSize(func);
        func->analysis.complexity = calculateComplexity(func);
        func->analysis.side_effects = analyzeSideEffects(func);
        
        // Mark as inline candidate if appropriate
        if (isInlineCandidate(func)) {
            func->optimization_flags |= INLINE_CANDIDATE;
        }
        
        return func;
    }
};
```



# LOCAL OPTIMIZATIONS

These optimizations work within a single basic block of code.

1. Constant Folding
This is one of the most fundamental optimizations. Here's how modern compilers implement it:

```cpp
class ConstantFolder {
    Value* fold(Expression* expr) {
        if (expr->isConstant()) {
            return expr->evaluate();
        }
        
        if (isBinaryOp(expr)) {
            Value* left = fold(expr->left);
            Value* right = fold(expr->right);
            
            if (left->isConstant() && right->isConstant()) {
                switch (expr->operator_type) {
                    case ADD:
                        return new ConstantValue(left->value + right->value);
                    case MULTIPLY:
                        return new ConstantValue(left->value * right->value);
                    // ... other operators
                }
            }
        }
        return expr;
    }
};
```

2. Strength Reduction
This replaces expensive operations with cheaper ones. Modern compilers like LLVM implement this like:

```cpp
class StrengthReducer {
    Expression* reduce(Expression* expr) {
        if (expr->type == MULTIPLY) {
            if (isPowerOfTwo(expr->right)) {
                // Convert multiplication by 2^n to left shift by n
                int shift = log2(expr->right->value);
                return new ShiftExpression(expr->left, shift);
            }
        }
        
        if (expr->type == DIVIDE) {
            if (isPowerOfTwo(expr->right)) {
                // Convert division by 2^n to right shift by n
                int shift = log2(expr->right->value);
                return new ShiftExpression(expr->left, -shift);
            }
        }
        return expr;
    }
};
```

LOOP OPTIMIZATIONS

These are more complex because they deal with program flow. Let's look at how modern compilers implement them.

1. Loop Invariant Code Motion (LICM)
LLVM implements this optimization to move code that doesn't change in a loop outside of it:

```cpp
class LoopInvariantMotion {
    void optimize(Loop* loop) {
        // Find all expressions in the loop
        for (Instruction* inst : loop->instructions()) {
            if (isInvariant(inst, loop)) {
                // Move to loop preheader
                moveToPreheader(inst, loop);
            }
        }
    }
    
    bool isInvariant(Instruction* inst, Loop* loop) {
        // An instruction is invariant if all its operands are
        // either defined outside the loop or are themselves invariant
        for (Value* operand : inst->operands()) {
            if (loop->contains(operand) && !isInvariant(operand, loop)) {
                return false;
            }
        }
        return !inst->hasMemoryEffects();
    }
};
```

2. Loop Unrolling
This optimization duplicates loop bodies to reduce branch overhead:

```cpp
class LoopUnroller {
    void unroll(Loop* loop, int factor) {
        if (!isUnrollCandidate(loop)) return;
        
        BasicBlock* body = loop->getBody();
        BasicBlock* header = loop->getHeader();
        
        // Create unrolled copies
        for (int i = 0; i < factor-1; i++) {
            BasicBlock* copy = cloneBasicBlock(body);
            // Update phi nodes and branch targets
            updatePhiNodes(copy, i);
            linkBlocks(header, copy);
        }
        
        // Adjust trip count
        updateTripCount(loop, factor);
    }
    
    bool isUnrollCandidate(Loop* loop) {
        // Check if loop has a known trip count
        // and contains no complex control flow
        return loop->getTripCount() && 
               !loop->hasComplexFlow();
    }
};
```

DATAFLOW OPTIMIZATIONS

These optimizations analyze how data moves through the program.

1. Common Subexpression Elimination (CSE)
Modern compilers implement this to avoid redundant calculations:

```cpp
class CSEOptimizer {
    void optimize(Function* func) {
        // Map expressions to their computed values
        HashMap<Expression*, Value*> computedValues;
        
        for (BasicBlock* block : func->blocks()) {
            for (Instruction* inst : block->instructions()) {
                Expression* expr = inst->asExpression();
                if (expr) {
                    if (Value* existing = computedValues[expr]) {
                        // Replace this computation with the existing value
                        inst->replaceAllUsesWith(existing);
                        inst->eraseFromParent();
                    } else {
                        computedValues[expr] = inst;
                    }
                }
            }
        }
    }
};
```

2. Dead Code Elimination (DCE)
This removes code that doesn't affect the program's output:

```cpp
class DeadCodeEliminator {
    void eliminate(Function* func) {
        // Mark all instructions as potentially dead
        Set<Instruction*> worklist;
        for (Instruction* inst : func->instructions()) {
            if (!inst->mayHaveSideEffects()) {
                worklist.insert(inst);
            }
        }
        
        // Keep instructions that affect output
        while (!worklist.empty()) {
            Instruction* inst = worklist.pop();
            if (isLive(inst)) {
                // Mark all operands as needed
                for (Value* operand : inst->operands()) {
                    if (Instruction* def = operand->definingInstruction()) {
                        worklist.insert(def);
                    }
                }
            } else {
                inst->eraseFromParent();
            }
        }
    }
    
    bool isLive(Instruction* inst) {
        // An instruction is live if it:
        // 1. Has side effects
        // 2. Computes a value used by other live instructions
        // 3. Affects control flow
        return inst->hasMemoryEffects() ||
               inst->hasUsers() ||
               inst->isTerminator();
    }
};
```

