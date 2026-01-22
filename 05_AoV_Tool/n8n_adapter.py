import argparse
import os
import cv2
import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Optional, List, Dict

# Import existing modules
# Assuming this script is placed in the root directory: D:\NKUST_LAB_Work_Data\Lab work\cv-algorithm-study\05_AoV_Tool
from logic_engine import LogicEngine
from processor import ImageProcessor

@dataclass
class AoVConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    image_path: str = ""
    output_path: str = "output.png"
    user_query: str = ""
    use_mock_llm: bool = False

class AoVTool:
    def __init__(self, config: AoVConfig):
        self.config = config
        
        # Initialize Core Components
        self.logic_engine = LogicEngine()
        self.processor = ImageProcessor()
        
        # Configure LLM if API Key is provided
        if self.config.api_key:
            # print(f"[AoVTool] Configuring LLM with API Key...")
            self.logic_engine.prompt_master.api_key = self.config.api_key
            self.logic_engine.prompt_master.base_url = self.config.base_url
            self.logic_engine.prompt_master.llm_available = True
            
            # Important: If key is provided, default to NOT using mock unless explicitly requested
            if not self.config.use_mock_llm:
                pass
                # print("[AoVTool] Mock LLM disabled. Using Real LLM.")
        else:
            # print("[AoVTool] No API Key provided. Force enabling Mock LLM.")
            self.config.use_mock_llm = True

    def run(self) -> str:
        # 1. Validate Input
        if not os.path.exists(self.config.image_path):
            raise FileNotFoundError(f"Input image not found: {self.config.image_path}")
            
        # print(f"[AoVTool] Loading image: {self.config.image_path}")
        image = cv2.imread(self.config.image_path)
        if image is None:
             raise ValueError(f"Failed to load image: {self.config.image_path}")

        # 2. Generate Pipeline via Logic Engine
        # print(f"[AoVTool] Processing Query: '{self.config.user_query}'")
        llm_result = self.logic_engine.process_user_query(
            self.config.user_query, 
            use_mock_llm=self.config.use_mock_llm
        )
        
        # [Refactor Fix] Handle Dict return type from LogicEngine
        if llm_result.get("error"):
            raise RuntimeError(f"AI Generation Failed: {llm_result['error']} (Reasoning: {llm_result.get('reasoning')})")
            
        pipeline = llm_result["pipeline"]
        reasoning = llm_result.get("reasoning", "No reasoning provided")
        
        print(f"[AoVTool] AI Reasoning: {reasoning}")
        
        # 3. Execute Pipeline via Processor
        # print(f"[AoVTool] Executing Pipeline with {len(pipeline)} nodes...")
        processed_image = self.processor.execute_pipeline(image, pipeline)
        
        # 4. Save Output
        # print(f"[AoVTool] Saving result to: {self.config.output_path}")
        cv2.imwrite(self.config.output_path, processed_image)
        
        return self.config.output_path

def main():
    parser = argparse.ArgumentParser(description="NKUST AoV Tool - n8n Adapter")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--output", default="output.png", help="Path to output image")
    parser.add_argument("--api_key", help="OpenAI API Key")
    parser.add_argument("--base_url", default="https://api.openai.com/v1", help="LLM Base URL")
    parser.add_argument("--mock", action="store_true", help="Force use mock LLM")
    
    args = parser.parse_args()
    
    config = AoVConfig(
        api_key=args.api_key if args.api_key else "",
        base_url=args.base_url,
        image_path=args.image,
        output_path=args.output,
        user_query=args.query,
        use_mock_llm=args.mock
    )
    
    # Capture stdout/stderr to prevent polluting n8n JSON output
    f = io.StringIO()
    # We keep the original stdout just in case we need to print the final JSON to it
    original_stdout = sys.stdout
    
    try:
        with redirect_stdout(f), redirect_stderr(f):
            tool = AoVTool(config)
            result_path = tool.run()
            
        # Success output - Print ONLY the JSON to original stdout
        print(json.dumps({
            "status": "success", 
            "output_path": os.path.abspath(result_path),
            "query": args.query,
            "used_mock": config.use_mock_llm,
            "logs": f.getvalue() 
        }))
        
    except Exception as e:
        # Restore stdout to print error JSON
        sys.stdout = original_stdout
        print(json.dumps({
            "status": "error", 
            "message": str(e),
            "logs": f.getvalue()
        }))
        exit(1)

if __name__ == "__main__":
    main()
