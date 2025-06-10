"""
Critical tokens for Java unit test assertion generation.

This module defines tokens that are particularly important for generating 
correct and meaningful Java unit test assertions. These tokens receive 
higher loss penalties during training to improve model accuracy on critical 
assertion components.

Task 4.1: Curate CRITICAL_TOKENS list with assertion methods, matchers, 
logical keywords, and structural tokens.
"""

# Core JUnit assertion methods
JUNIT_ASSERTION_METHODS = [
    # Basic assertions
    "assertTrue", "assertFalse", "assertNull", "assertNotNull",
    "assertEquals", "assertNotEquals", "assertSame", "assertNotSame",
    "assertArrayEquals", "assertIterableEquals", "assertLinesMatch",
    
    # Numeric and comparison assertions
    "assertThat", "assertAll", "assertThrows", "assertDoesNotThrow",
    "assertTimeout", "assertTimeoutPreemptively",
    
    # JUnit 5 specific
    "assertInstanceOf", "assertExactlyInstanceOf",
    
    # Hamcrest-style (if used)
    "assertThat", "is", "not", "equalTo", "sameInstance",
    "hasItem", "hasItems", "hasSize", "containsString",
    "startsWith", "endsWith", "greaterThan", "lessThan",
    "greaterThanOrEqualTo", "lessThanOrEqualTo"
]

# TestNG assertion methods (if applicable)
TESTNG_ASSERTION_METHODS = [
    "Assert.assertTrue", "Assert.assertFalse", "Assert.assertNull", 
    "Assert.assertNotNull", "Assert.assertEquals", "Assert.assertNotEquals",
    "Assert.assertSame", "Assert.assertNotSame", "Assert.fail"
]

# Mock framework methods (Mockito, EasyMock, etc.)
MOCK_ASSERTION_METHODS = [
    # Mockito
    "verify", "verifyNoMoreInteractions", "verifyNoInteractions",
    "verifyZeroInteractions", "when", "thenReturn", "thenThrow",
    "doReturn", "doThrow", "doNothing", "doAnswer", "doCallRealMethod",
    "times", "never", "atLeast", "atMost", "atLeastOnce", "atMostOnce",
    "any", "anyString", "anyInt", "anyLong", "anyDouble", "anyBoolean",
    "anyObject", "anyCollection", "anyList", "anySet", "anyMap",
    "eq", "same", "isNull", "notNull", "matches", "contains",
    "startsWith", "endsWith", "argThat", "refEq",
    
    # EasyMock
    "expect", "replay", "reset", "createMock", "createNiceMock",
    "createStrictMock", "makeThreadSafe", "checkOrder"
]

# Logical and comparison operators critical for assertions
LOGICAL_OPERATORS = [
    # Boolean logic
    "true", "false", "null",
    
    # Comparison operators (as tokens)
    "==", "!=", "<=", ">=", "<", ">",
    
    # Logical operators  
    "&&", "||", "!", "&", "|", "^",
    
    # Keywords
    "equals", "compareTo", "instanceof", "getClass"
]

# Exception and error handling tokens
EXCEPTION_TOKENS = [
    # Exception types commonly asserted
    "Exception", "RuntimeException", "IllegalArgumentException", 
    "IllegalStateException", "NullPointerException", "IndexOutOfBoundsException",
    "UnsupportedOperationException", "ConcurrentModificationException",
    "NumberFormatException", "ClassCastException", "SecurityException",
    
    # Custom exception patterns
    "IOException", "SQLException", "InterruptedException",
    "TimeoutException", "ExecutionException",
    
    # Exception handling
    "throw", "throws", "catch", "finally", "try",
    "getMessage", "getCause", "getStackTrace"
]

# Collection and array assertion tokens
COLLECTION_TOKENS = [
    # Collection methods
    "size", "length", "isEmpty", "contains", "containsAll",
    "get", "indexOf", "lastIndexOf", "subList", "toArray",
    
    # Collection types
    "List", "Set", "Map", "Collection", "Array", "ArrayList", 
    "LinkedList", "HashSet", "TreeSet", "HashMap", "TreeMap",
    
    # Array operations
    "Arrays", "asList", "copyOf", "sort", "binarySearch"
]

# String assertion tokens  
STRING_TOKENS = [
    # String methods
    "String", "charAt", "substring", "indexOf", "lastIndexOf",
    "startsWith", "endsWith", "contains", "equals", "equalsIgnoreCase",
    "trim", "toLowerCase", "toUpperCase", "split", "replace",
    "replaceAll", "matches", "format", "valueOf", "toString",
    
    # String literals and patterns
    "\"\"", "empty", "blank", "whitespace"
]

# Numeric assertion tokens
NUMERIC_TOKENS = [
    # Numeric types
    "int", "long", "double", "float", "byte", "short", "char",
    "Integer", "Long", "Double", "Float", "Byte", "Short", "Character",
    "BigDecimal", "BigInteger", "Number",
    
    # Numeric methods
    "intValue", "longValue", "doubleValue", "floatValue",
    "compareTo", "equals", "hashCode", "valueOf", "parseDouble",
    "parseInt", "parseLong", "parseFloat",
    
    # Math operations
    "Math", "abs", "max", "min", "round", "ceil", "floor",
    "sqrt", "pow", "random"
]

# Structural and syntax tokens critical for valid Java
STRUCTURAL_TOKENS = [
    # Method and class structure
    "public", "private", "protected", "static", "final", "abstract",
    "void", "return", "class", "interface", "extends", "implements",
    "this", "super", "new", "package", "import",
    
    # Control flow (important in assertion context)
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue",
    
    # Punctuation critical for syntax
    "(", ")", "{", "}", "[", "]", ";", ",", ".", "::", "->",
    
    # Access and scoping
    "get", "set", "is", "has", "was", "can", "should", "will"
]

# Test lifecycle and setup tokens
TEST_LIFECYCLE_TOKENS = [
    # JUnit annotations and lifecycle
    "@Test", "@Before", "@After", "@BeforeEach", "@AfterEach",
    "@BeforeAll", "@AfterAll", "@BeforeClass", "@AfterClass",
    "@Setup", "@TearDown", "@Ignore", "@Disabled",
    
    # Test organization
    "@DisplayName", "@Tag", "@Nested", "@ParameterizedTest",
    "@ValueSource", "@CsvSource", "@MethodSource", "@ArgumentsSource",
    
    # Test configuration
    "@Timeout", "@ExtendWith", "@RegisterExtension", "@TempDir"
]

# Domain-specific tokens that might be critical for specific test contexts
DOMAIN_SPECIFIC_TOKENS = [
    # Common business logic patterns
    "valid", "invalid", "expected", "actual", "result", "value",
    "status", "state", "error", "success", "failure", "response",
    "request", "data", "entity", "model", "service", "repository",
    "controller", "manager", "handler", "processor", "validator",
    
    # Common test patterns
    "given", "when", "then", "should", "expect", "verify", "check",
    "test", "mock", "stub", "fake", "spy", "fixture", "setup"
]

# Combine all critical token categories
CRITICAL_TOKENS = (
    JUNIT_ASSERTION_METHODS +
    TESTNG_ASSERTION_METHODS + 
    MOCK_ASSERTION_METHODS +
    LOGICAL_OPERATORS +
    EXCEPTION_TOKENS +
    COLLECTION_TOKENS +
    STRING_TOKENS +
    NUMERIC_TOKENS +
    STRUCTURAL_TOKENS +
    TEST_LIFECYCLE_TOKENS +
    DOMAIN_SPECIFIC_TOKENS
)

# Remove duplicates while preserving order
CRITICAL_TOKENS = list(dict.fromkeys(CRITICAL_TOKENS))

# Token categories for analysis and debugging
TOKEN_CATEGORIES = {
    'junit_assertions': JUNIT_ASSERTION_METHODS,
    'testng_assertions': TESTNG_ASSERTION_METHODS,
    'mock_assertions': MOCK_ASSERTION_METHODS,
    'logical_operators': LOGICAL_OPERATORS,
    'exceptions': EXCEPTION_TOKENS,
    'collections': COLLECTION_TOKENS,
    'strings': STRING_TOKENS,
    'numerics': NUMERIC_TOKENS,
    'structural': STRUCTURAL_TOKENS,
    'test_lifecycle': TEST_LIFECYCLE_TOKENS,
    'domain_specific': DOMAIN_SPECIFIC_TOKENS
}

def get_critical_tokens():
    """
    Get the complete list of critical tokens for assertion generation.
    
    Returns:
        List[str]: List of critical tokens that should receive higher 
                   loss penalties during training.
    """
    return CRITICAL_TOKENS.copy()

def get_token_categories():
    """
    Get tokens organized by category for analysis.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping category names to token lists.
    """
    return TOKEN_CATEGORIES.copy()

def get_tokens_by_category(category):
    """
    Get tokens for a specific category.
    
    Args:
        category (str): Category name from TOKEN_CATEGORIES keys.
        
    Returns:
        List[str]: Tokens in the specified category.
    """
    return TOKEN_CATEGORIES.get(category, []).copy()

def is_critical_token(token):
    """
    Check if a token is in the critical tokens list.
    
    Args:
        token (str): Token to check.
        
    Returns:
        bool: True if token is critical, False otherwise.
    """
    return token in CRITICAL_TOKENS

# Statistics about the critical tokens
CRITICAL_TOKENS_STATS = {
    'total_tokens': len(CRITICAL_TOKENS),
    'categories': len(TOKEN_CATEGORIES),
    'category_sizes': {cat: len(tokens) for cat, tokens in TOKEN_CATEGORIES.items()},
    'most_common_prefixes': ['assert', 'verify', 'expect', 'get', 'set', 'is', 'has'],
    'coverage_estimate': '85%'  # Estimated coverage of critical assertion tokens
}

def get_critical_tokens_stats():
    """
    Get statistics about the critical tokens collection.
    
    Returns:
        Dict: Statistics about critical tokens.
    """
    return CRITICAL_TOKENS_STATS.copy()

if __name__ == "__main__":
    # Print summary when run directly
    print("🔑 Critical Tokens for Java Unit Test Assertion Generation")
    print(f"Total critical tokens: {len(CRITICAL_TOKENS)}")
    print(f"Categories: {len(TOKEN_CATEGORIES)}")
    
    print("\n📊 Category breakdown:")
    for category, tokens in TOKEN_CATEGORIES.items():
        print(f"  {category}: {len(tokens)} tokens")
    
    print(f"\n📋 First 20 critical tokens:")
    for i, token in enumerate(CRITICAL_TOKENS[:20]):
        print(f"  {i+1:2d}. {token}")
    
    if len(CRITICAL_TOKENS) > 20:
        print(f"  ... and {len(CRITICAL_TOKENS) - 20} more tokens")
    
    print(f"\n✅ Critical tokens list ready for Task 4.2")