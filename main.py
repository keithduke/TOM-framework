#!/usr/bin/env python3
"""
T.O.M. CLI - Interactive AI Assistant
Entry point with argument parsing
"""

import sys
from pathlib import Path
from cli import ChatInterface, clear_cache_file


def main():
    """Parse arguments and start chat interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="T.O.M. CLI - Interactive AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --model ./path/to/model
  python main.py --no-cache --debug
  python main.py clear-cache --force
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # clear-cache subcommand
    clear_parser = subparsers.add_parser('clear-cache', help='Clear cache file')
    clear_parser.add_argument('--model', '-m', default='./Qwen3-4B-Thinking-2507-8bit',
                             help='Model path')
    clear_parser.add_argument('--cache', '-c', help='Cache file path')
    clear_parser.add_argument('--force', '-f', action='store_true',
                             help='Delete without confirmation')
    
    # Main arguments
    parser.add_argument('--model', '-m', default='./Qwen3-4B-Thinking-2507-8bit',
                       help='Path to MLX model')
    parser.add_argument('--cache', '-c', help='Cache file path')
    parser.add_argument('--max-context', type=int, help='Override max context tokens')
    parser.add_argument('--gc-frequency', type=int, default=3,
                       help='GC frequency (every N generations)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--no-prewarm', action='store_true', help='Skip cache prewarm')
    parser.add_argument('--no-auto-gc', action='store_true', help='Disable auto GC')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Handle clear-cache
    if args.command == 'clear-cache':
        clear_cache_file(args.model, args.cache, args.force)
        return
    
    # Set logging
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    # Create and run interface
    chat = ChatInterface(
        model_path=model_path,
        cache_path=args.cache,
        enable_cache=not args.no_cache,
        prewarm=not args.no_prewarm,
        max_context_override=args.max_context,
        auto_gc=not args.no_auto_gc,
        gc_frequency=args.gc_frequency
    )
    
    chat.load_model()
    chat.run()


if __name__ == "__main__":
    main()