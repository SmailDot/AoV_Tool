import os
import cv2
import json
import time
import base64
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import Core Components
from app.core.logic_engine import LogicEngine
from app.core.processor import ImageProcessor
from app.core.code_generator import CodeGenerator

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov', 'mkv'}

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

def get_file_type(filename):
    if '.' not in filename: return None
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS_IMG: return 'image'
    if ext in ALLOWED_EXTENSIONS_VID: return 'video'
    return None

def decode_image(base64_string):
    """Convert base64 string to numpy array"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image_bgr):
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/', methods=['GET'])
def index():
    """Root endpoint for sanity checks"""
    return jsonify({
        "status": "running",
        "service": "NKUST AoV Tool API",
        "endpoints": {
            "health": "GET /health",
            "process": "POST /process",
            "algorithms": "GET /algorithms"
        },
        "message": "Use /process endpoint for n8n integration."
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "NKUST AoV Tool API",
        "version": "1.0"
    })

@app.route('/process', methods=['GET', 'POST'])
def process_pipeline():
    """
    Main processing endpoint for n8n.
    Accepts: JSON with 'query' and ('image_path' OR 'image_base64')
    Returns: JSON with result path and metadata
    """
    if request.method == 'GET':
        return jsonify({
            "status": "error",
            "message": "Method Not Allowed for GET. Please use POST request with JSON body.",
            "example_body": {
                "query": "Detect coins",
                "image_path": "C:/path/to/image.jpg"
            }
        }), 405

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

        # 2. Identify Input Type & Load
        input_type = "unknown"
        input_source = "unknown"
        valid_input = False
        image = None # Init to prevent unbound local error
        execution_mode = data.get('execution_mode', 'full') # Default to full execution
        
        # If plan_only, we SKIP file validation entirely
        if execution_mode == 'plan_only':
            input_type = "none"
            valid_input = True
        else:
            # Only validate files in 'full' mode
            if image_path:
                if not os.path.exists(image_path):
                    return jsonify({"error": f"File path not found: {image_path}"}), 404
                
                input_source = image_path
                file_type = get_file_type(image_path)
                
                if file_type == 'image':
                    input_type = 'image'
                    image = cv2.imread(image_path)
                    valid_input = True
                elif file_type == 'video':
                    input_type = 'video'
                    image = None # Video doesn't load whole file into memory
                    valid_input = True
                else:
                    return jsonify({"error": "Unsupported file extension"}), 400
                
            elif image_base64:
                # Base64 currently only supports images for simplicity
                try:
                    image = decode_image(image_base64)
                    input_source = "base64_input"
                    input_type = 'image'
                    valid_input = True
                except Exception as e:
                    return jsonify({"error": f"Invalid base64 string: {str(e)}"}), 400
            else:
                return jsonify({"error": "Must provide 'image_path' or 'image_base64' (unless execution_mode='plan_only')"}), 400

        # Safety check for image type (only if executing)
        if execution_mode != 'plan_only' and input_type == 'image' and image is None:
             return jsonify({"error": "Failed to decode image content"}), 500

        # 3. Configure LLM
        if api_key:
            logic_engine.prompt_master.api_key = api_key
            logic_engine.prompt_master.llm_available = True
        
        # 4. Generate Pipeline (Brain)
        print(f"[Server] Processing Query: {user_query} (Mode: {execution_mode})")
        llm_result = logic_engine.process_user_query(user_query, use_mock_llm=use_mock)
        
        if llm_result.get("error"):
            return jsonify({
                "status": "error",
                "error": llm_result["error"],
                "reasoning": llm_result.get("reasoning")
            }), 500
            
        pipeline = llm_result["pipeline"]
        
        # 5. Execute Pipeline (Muscle) - Skip if plan_only
        timestamp = int(time.time())
        execution_stats = {}
        output_path = "" # Init for scope safety
        result_base64 = None # Init for scope safety
        
        if execution_mode != 'plan_only':
            if input_type == 'image':
                if image is None: # Double check (though caught above)
                    return jsonify({"error": "Image data lost"}), 500

                print(f"[Server] Executing Image Pipeline ({len(pipeline)} nodes)...")
                processed_image = processor.execute_pipeline(image, pipeline)
                
                output_filename = f"result_{timestamp}.png"
                output_path = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], output_filename))
                cv2.imwrite(output_path, processed_image)
                
                # Base64 output logic (Image only)
                if data.get('return_base64'):
                    result_base64 = encode_image(processed_image)
                    
            elif input_type == 'video':
                print(f"[Server] Executing Video Pipeline ({len(pipeline)} nodes)...")
                output_filename = f"result_{timestamp}.mp4"
                output_path = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], output_filename))
                
                # Run video processing
                video_stats = processor.process_video(input_source, output_path, pipeline)
                execution_stats.update(video_stats)
                result_base64 = None # Don't return base64 for video (too large)
        
        # 6. Generate Code
        generated_python_code = CodeGenerator.generate_python_script(pipeline)

        execution_time = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "type": input_type,
            "query": user_query,
            "reasoning": llm_result.get("reasoning"),
            "pipeline_summary": [n['name'] for n in pipeline],
            "pipeline_json": pipeline, # [NEW] Return full pipeline JSON
            "input_source": input_source,
            "output_path": output_path,
            "output_base64": result_base64,
            "execution_time_sec": round(execution_time, 2),
            "video_stats": execution_stats if input_type == 'video' else None,
            "fpga_estimated_clk": sum(n['fpga_constraints']['estimated_clk'] for n in pipeline),
            "generated_code": generated_python_code
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

def start_ngrok(port: int) -> str | None:
    """Start ngrok tunnel and return public URL"""
    try:
        from pyngrok import ngrok
        
        # Check if authtoken is configured
        # You can set it via: ngrok config add-authtoken <token>
        # Or set environment variable: NGROK_AUTHTOKEN
        
        public_url = ngrok.connect(str(port), bind_tls=True).public_url
        print("=" * 60)
        print("🌐 NGROK TUNNEL ACTIVE")
        print("=" * 60)
        print(f"📡 Public URL: {public_url}")
        print(f"📡 Health Check: {public_url}/health")
        print(f"📡 Process API: {public_url}/process")
        print("=" * 60)
        return public_url
    except ImportError:
        print("[Warning] pyngrok not installed. Run: pip install pyngrok")
        print("[Warning] Server will only be accessible locally.")
        return None
    except Exception as e:
        print(f"[Warning] ngrok failed to start: {e}")
        print("[Warning] Server will only be accessible locally.")
        return None


if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from n8n (if in docker)
    port = int(os.environ.get('PORT', 5000))
    
    # Start ngrok tunnel for remote access
    use_ngrok = os.environ.get('USE_NGROK', 'true').lower() == 'true'
    if use_ngrok:
        public_url = start_ngrok(port)
    
    print(f"[Server] Starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
