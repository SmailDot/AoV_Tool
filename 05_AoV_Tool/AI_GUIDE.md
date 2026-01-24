# AI_GUIDE.md - Agent Handoff Protocol

> **Version**: 2.0 (Knowledge-Driven Edition)
> **Last Updated**: 2026-01-24
> **Status**: Production Ready

## 1. Project Identity
**Name**: NKUST AoV Tool (FPGA-Aware Vision Pipeline Generator)
**Goal**: A "Teacher-Student" system that evolves from a tool into a knowledge base. It allows users to generate, optimize, and store Computer Vision pipelines for FPGA deployment.

## 2. Core Architecture

### Frontend (UI)
- **File**: `aov_app.py` (Streamlit)
- **Role**: The command center. Handles image upload, pipeline editing, and visualization.
- **Key Components**:
  - `components/node_editor.py`: Parameter editing (Auto-generated UI).
  - `components/visualizer.py`: Graphviz pipeline visualization.

### Backend (Logic & Processing)
- **Brain**: `logic_engine.py` (LLM Orchestration & Prompt Engineering).
- **Muscle**: `processor.py` (OpenCV Execution Engine).
- **Database**: `library_manager.py` (Manages `tech_lib.json` - Algorithm definitions).

### Intelligence Modules (The "Special Sauce")
1.  **AutoTuner (app/vision/optimizer)**:
    -   **Goal**: Automatically tune parameters to match a ground truth mask.
    -   **Tech**: Genetic Algorithm (GA) with Structural Mutation + Heuristic Expert Logic.
    -   **Feature**: "LLM Vision Feedback" (Simulated or Real) to diagnose image issues (too dark, too noisy) and suggest fixes.
    -   **Fixes Applied**: Smart Resize protection (anti-flattening), Aspect Ratio checks.

2.  **Knowledge Base (app/knowledge)**:
    -   **Goal**: "The Master's Notebook" - Store and retrieve successful experiences.
    -   **Tech**: **Multimodal RAG** (CLIP + FAISS).
    -   **Flow**: Image -> CLIP Embedding -> FAISS Vector Search -> Top-k Similar Cases.
    -   **Files**: `base.py` (Core Logic), `knowledge_db.json` (Metadata).

## 3. Key Features & Workflows

### A. Pipeline Generation (LLM)
- User types "Detect coins" -> LLM generates JSON pipeline -> Loaded into Editor.

### B. Auto-Tuning (Optimization)
- **Input**: Source Image + Target Mask (Binary).
- **Process**: 
    -   GA tries to mutate parameters (Hill Climbing).
    -   Structural Mutation tries to Add/Remove nodes (e.g., adding `Dilate` to fill holes).
    -   **Safety**: Resizes are locked to aspect ratio to prevent distortion.

### C. Knowledge Retrieval (Smart Suggest)
- **Input**: New Image.
- **Action**: Click "Smart Suggest".
- **Backend**: CLIP extracts visual features -> Searches FAISS DB.
- **Output**: Recommends pipelines from similar historical cases (e.g., "This looks like the 'Metal Scratch' case").

## 4. Maintenance & known Issues

### File Structure
- `app/`: Core application logic (Knowledge, Vision, Engine).
- `components/`: UI sub-modules.
- `lib/`: External libraries or legacy code.
- `tech_lib.json`: **CRITICAL**. Defines all available nodes. If you add a new OpenCV function, you MUST add it here first.

### Common Pitfalls
1.  **Streamlit State**: `st.session_state` is volatile. Always ensure state sync when modifying backend objects (like AutoTuner does).
2.  **OpenCV Types**: `st.number_input` crashes if given `numpy.int`. Always cast to `int()` or `float()` in UI code.
3.  **Resize Distortion**: `cv2.resize` with fixed W/H destroys aspect ratio. Use the `op_resize` in `basic.py` which handles `0` as "Auto".

### Roadmap (Future Agents)
- [ ] **Cloud Storage**: Move `knowledge_db.json` and images to Google Cloud Storage (GCS) for team sharing.
- [ ] **Real LLM Feedback**: Connect a real GPT-4o-Vision key to `AutoTuner` to replace the "Simulated Expert".
- [ ] **FPGA Export**: Enhance `code_generator.py` to support more HLS pragmas.

---
*Handed off by Sisyphus Agent - The "Teacher-Student" Architect.*
