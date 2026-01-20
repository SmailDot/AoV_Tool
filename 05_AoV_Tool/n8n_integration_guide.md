# n8n AI Integration Guide

> **For n8n AI Agents & Workflow Builders**
> This document describes how to programmatically interact with the NKUST AoV Tool codebase.

## 1. Core Architecture for Automation
The system is designed to be modular. n8n workflows should interact with specific modules rather than the UI (`aov_app.py`).

| Module | Purpose | Key Method | Return Type |
|--------|---------|------------|-------------|
| `logic_engine.py` | **Planner**. Converts text to pipeline JSON. | `process_user_query(text)` | `List[Dict]` (Pipeline JSON) |
| `processor.py` | **Executor**. Runs OpenCV ops. | `execute_pipeline(img, json)` | `np.ndarray` (Image) |
| `processor.py` | **Introspection**. Lists capabilities. | `get_supported_operations()` | `Dict` (Schema) |
| `library_manager.py` | **Knowledge Base**. Manages algorithms. | `list_algorithms()` | `List[Dict]` |

## 2. Introspection (How to learn what I can do)
To avoid hallucinations, AI agents should query the `ImageProcessor` to understand available tools.

```python
from processor import ImageProcessor
proc = ImageProcessor()

# Returns a JSON schema of all supported operations and their descriptions
capabilities = proc.get_supported_operations()
print(json.dumps(capabilities, indent=2))
```

**Output Example:**
```json
{
  "GaussianBlur": {
    "description": "Apply Gaussian Blur to reduce noise.",
    "type": "computer_vision_operation"
  },
  "Canny": {
    "description": "Detect edges using Canny algorithm.",
    "type": "computer_vision_operation"
  }
}
```

## 3. Pipeline JSON Schema
The core data structure is the **Pipeline List**. AI agents should generate JSON matching this format:

```json
[
  {
    "id": "node_0",
    "function": "GaussianBlur",  // Must match a key in get_supported_operations()
    "parameters": {
      "ksize": {"default": [5, 5]},
      "sigmaX": {"default": 0}
    },
    "fpga_constraints": {
      "estimated_clk": 150,
      "resource_usage": "Medium"
    }
  }
]
```

## 4. Automation Workflow Example (Python Script)
An n8n `Execute Command` node can run a script like this to process images without the UI:

```python
import cv2
from processor import ImageProcessor
from logic_engine import LogicEngine

# 1. Initialize
engine = LogicEngine()
processor = ImageProcessor()

# 2. Plan (Text -> JSON)
pipeline = engine.process_user_query("Detect edges in this image")

# 3. Execute (Image + JSON -> Image)
img = cv2.imread("input.jpg")
result = processor.execute_pipeline(img, pipeline)

# 4. Save
cv2.imwrite("output.jpg", result)
```

## 5. File Structure for Context
- **`tech_lib.json`**: The Single Source of Truth for algorithm data. Read this to understand parameter constraints.
- **`processor.py`**: The execution logic. Read `operation_map` to see implementation details.
