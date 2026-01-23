import os
import cv2
import json
import time
import base64
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import Core Components
from logic_engine import LogicEngine
from processor import ImageProcessor

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Agents (Global persistence)
print("[Server] Initializing Logic Engine & Processor...")
logic_engine = LogicEngine()
processor = ImageProcessor()
print("[Server] Initialization Complete.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_image(base64_string):
    """Convert base64 string to numpy array"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image_bgr):
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "NKUST AoV Tool API",
        "version": "1.0"
    })

@app.route('/process', methods=['POST'])
def process_pipeline():
    """
    Main processing endpoint for n8n.
    Accepts: JSON with 'query' and ('image_path' OR 'image_base64')
    Returns: JSON with result path and metadata
    """
    try:
        start_time = time.time()
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # 1. Parse Inputs
        user_query = data.get('query')
        image_path = data.get('image_path')
        image_base64 = data.get('image_base64')
        api_key = data.get('api_key')
        use_mock = data.get('use_mock', False)
        
        if not user_query:
            return jsonify({"error": "Missing 'query' field"}), 400

        # 2. Load Image
        image = None
        input_source = "unknown"
        
        if image_path:
            if not os.path.exists(image_path):
                return jsonify({"error": f"Image path not found: {image_path}"}), 404
            image = cv2.imread(image_path)
            input_source = image_path
            
        elif image_base64:
            try:
                image = decode_image(image_base64)
                input_source = "base64_input"
            except Exception as e:
                return jsonify({"error": f"Invalid base64 string: {str(e)}"}), 400
                
        else:
            return jsonify({"error": "Must provide 'image_path' or 'image_base64'"}), 400

        if image is None:
            return jsonify({"error": "Failed to decode/load image"}), 500

        # 3. Configure LLM
        if api_key:
            logic_engine.prompt_master.api_key = api_key
            logic_engine.prompt_master.llm_available = True
        
        # 4. Generate Pipeline (Brain)
        print(f"[Server] Processing Query: {user_query}")
        llm_result = logic_engine.process_user_query(user_query, use_mock_llm=use_mock)
        
        if llm_result.get("error"):
            return jsonify({
                "status": "error",
                "error": llm_result["error"],
                "reasoning": llm_result.get("reasoning")
            }), 500
            
        pipeline = llm_result["pipeline"]
        
        # 5. Execute Pipeline (Muscle)
        print(f"[Server] Executing {len(pipeline)} nodes...")
        processed_image = processor.execute_pipeline(image, pipeline)
        
        # 6. Save Output
        timestamp = int(time.time())
        output_filename = f"result_{timestamp}.png"
        output_path = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], output_filename))
        
        cv2.imwrite(output_path, processed_image)
        
        # Optional: Return base64 if requested
        result_base64 = None
        if data.get('return_base64'):
            result_base64 = encode_image(processed_image)

        execution_time = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "query": user_query,
            "reasoning": llm_result.get("reasoning"),
            "pipeline_summary": [n['name'] for n in pipeline],
            "input_source": input_source,
            "output_path": output_path,
            "output_base64": result_base64, # Only if requested
            "execution_time_sec": round(execution_time, 2),
            "fpga_estimated_clk": sum(n['fpga_constraints']['estimated_clk'] for n in pipeline)
        })

    except Exception as e:
        print(f"[Server Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/algorithms', methods=['GET'])
def list_algorithms():
    """List all available algorithms for n8n dropdowns"""
    try:
        libs = logic_engine.lib_manager.data.get('libraries', {})
        algo_list = []
        
        for category, algos in libs.items():
            for key, val in algos.items():
                algo_list.append({
                    "id": key,
                    "name": val.get("name"),
                    "description": val.get("description"),
                    "category": category
                })
        
        return jsonify({"algorithms": algo_list, "count": len(algo_list)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from n8n (if in docker)
    port = int(os.environ.get('PORT', 5000))
    print(f"[Server] Starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
