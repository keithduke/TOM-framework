"""
Configuration constants for T.O.M. CLI
"""

# Generation parameters
MAX_GENERATION_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.0

# Context management
DEFAULT_MODEL_MAX_CONTEXT = 32768
CONTEXT_USAGE_RATIO = 0.8  # Use 80% of model's max context
CONTEXT_TRIM_TARGET_RATIO = 0.8  # Trim to 80% when exceeded
TOOL_RESULT_CONTEXT_RATIO = 0.2  # Max 20% of context for tool results
MAX_TOOL_RESULT_TOKENS = 8192

# Memory management
DEFAULT_GC_FREQUENCY = 3
LOW_MEMORY_THRESHOLD_GB = 2.0

# File reading limits
MAX_FILE_SIZE_MB = 10

# System prompt
DEFAULT_SYSTEM_PROMPT = "You are here."
