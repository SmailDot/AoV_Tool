# n8n AI Integration Guide

> **For n8n Workflow Builders**
> This document describes how to integrate the NKUST AoV Tool into your n8n workflows using the unified adapter.

## 1. Quick Start (The Adapter Way)

The easiest way to use this tool in n8n is via the **Execute Command** node calling `n8n_adapter.py`. This adapter handles:
- LLM interaction (OpenAI/Mock)
- Image I/O (OpenCV)
- JSON formatting for n8n (handling stdout/stderr cleanly)

### n8n Node Configuration
- **Node Type**: Execute Command
- **Command**:
```bash
python "D:\NKUST_LAB_Work_Data\Lab work\cv-algorithm-study\05_AoV_Tool\n8n_adapter.py" \
  --image "/path/to/input.jpg" \
  --query "Detect coins and calculate their value" \
  --output "/path/to/output.png" \
  --api_key "sk-..." 
```

### Arguments
| Flag | Description | Required |
|------|-------------|----------|
| `--image` | Absolute path to input image | Yes |
| `--query` | Natural language description of task | Yes |
| `--output` | Path to save processed image | No (Default: output.png) |
| `--api_key` | OpenAI API Key. If omitted, tool forces Mock Mode. | No |
| `--base_url` | Custom LLM URL (e.g. for local models) | No |
| `--mock` | Force Mock Mode even if API Key is present | No |

## 2. JSON Output Structure

The adapter returns a pure JSON object to stdout, ready for n8n parsing.

```json
{
  "status": "success",
  "output_path": "D:\\...\\result.png",
  "query": "detect coins",
  "used_mock": false,
  "logs": "[AI Reasoning] Since the image is noisy, I suggest using the advanced detector..."
}
```

## 3. Advanced: Architecture (Class-Based)

If you are writing a custom Python script node in n8n instead of using the CLI adapter, use the `AoVTool` class.

```python
from n8n_adapter import AoVTool, AoVConfig

# 1. Configure
config = AoVConfig(
    api_key="sk-...",
    image_path="input.jpg",
    user_query="detect circles",
    output_path="result.png"
)

# 2. Initialize Tool
tool = AoVTool(config)

# 3. Run
try:
    result_path = tool.run()
    print(f"Success: {result_path}")
except Exception as e:
    print(f"Error: {e}")
```

## 4. Supported Algorithms

The tool supports 30+ algorithms including:
- **Basic**: GaussianBlur, Canny, Dilate, Erode
- **Advanced**: HoughCircles, FindContours, OpticalFlow
- **Specialized**: `advanced_coin_detection` (Automatically triggered for coin tasks)

See `tech_lib.json` for full list and parameter constraints.
