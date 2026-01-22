
import json
import sys

try:
    with open('tech_lib.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("✅ JSON format is VALID.")
except json.JSONDecodeError as e:
    print(f"❌ JSON Error at line {e.lineno}, column {e.colno}:")
    print(f"   {e.msg}")
except Exception as e:
    print(f"❌ Other Error: {e}")
