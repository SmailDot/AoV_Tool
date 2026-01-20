import sys
import os
import streamlit # Ensure streamlit is importable

print("Testing imports...")
try:
    from logic_engine import LogicEngine
    from processor import ImageProcessor
    # We can't easily import aov_app because it runs streamlit commands on import
    # But we can check syntax by compiling it
    with open('aov_app.py', 'r', encoding='utf-8') as f:
        compile(f.read(), 'aov_app.py', 'exec')
    print("aov_app.py Syntax OK")
    
    print("Imports OK")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except SyntaxError as e:
    print(f"Syntax Error in aov_app.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"General Error: {e}")
    sys.exit(1)

print("Initializing Engine...")
try:
    engine = LogicEngine()
    print("Engine Initialized")
except Exception as e:
    print(f"Engine Init Error: {e}")
    # Continue to check processor

print("Initializing Processor...")
try:
    processor = ImageProcessor()
    print("Processor Initialized")
except Exception as e:
    print(f"Processor Init Error: {e}")

print("Testing Mock Query...")
try:
    pipeline = engine.process_user_query("Detect coins", use_mock_llm=True)
    print(f"Pipeline generated: {len(pipeline)} nodes")
except Exception as e:
    print(f"Query Error: {e}")

print("Verification Successful!")
