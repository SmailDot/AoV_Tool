# AI_GUIDE.md - Agent Handoff Protocol

> **Version**: 3.0 (Tech-UI Edition + Complete Feature Set)
> **Last Updated**: 2026-01-31
> **Status**: Production Ready with Tech Theme

## 1. Project Identity
**Name**: NKUST AoV Tool (FPGA-Aware Vision Pipeline Generator)
**Goal**: A "Teacher-Student" system that evolves from a tool into a knowledge base. It allows users to generate, optimize, and store Computer Vision pipelines for FPGA deployment.

## 2. Core Architecture

### Frontend (UI)
- **File**: `aov_app.py` (Streamlit)
- **Role**: The command center with **Cyberpunk/Tech Theme**.
- **Key Components**:
  - `components/node_editor.py`: Parameter editing (Auto-generated UI).
  - `components/visualizer.py`: Graphviz pipeline visualization.
  - `components/knowledge_tree.py`: D3.js animated tree visualization for knowledge base exploration.
  - `components/sidebar.py`: Settings and navigation.
  - `components/style.py`: **[Updated]** Tech Theme CSS with neon glows, animations, and mouse effects.

### Backend (Logic & Processing)
- **Brain**: `logic_engine.py` (LLM Orchestration & Prompt Engineering).
- **Muscle**: `processor.py` (OpenCV Execution Engine).
  - **[New]** `_normalize_func_name()`: Auto-converts cv2.GaussianBlur → gaussian_blur.
- **Database**: `library_manager.py` (Manages `tech_lib.json` - Algorithm definitions).

### Intelligence Modules
1.  **AutoTuner (app/vision/optimizer)**:
    -   **Goal**: Automatically tune parameters to match a ground truth mask.
    -   **Tech**: Genetic Algorithm (GA) with Structural Mutation + Heuristic Expert Logic.
    -   **Feature**: "LLM Vision Feedback" to diagnose image issues.
    -   **Fixes Applied**: Smart Resize protection, Aspect Ratio checks.

2.  **Knowledge Base (app/knowledge)**:
    -   **Goal**: "The Master's Notebook" - Store and retrieve successful experiences.
    -   **Tech**: Multimodal RAG (CLIP + FAISS).
    -   **Flow**: Image → CLIP Embedding → FAISS Vector Search → Top-k Similar Cases.
    -   **Files**: `base.py` (Core Logic), `knowledge_db.json` (Metadata).
    -   **Features**:
        -   **Text Search**: Natural language query.
        -   **Image Search**: Visual similarity (CLIP).
        -   **Knowledge Tree**: Interactive D3.js visualization.
        -   **[New]** Save to Folder now auto-adds to Knowledge Base with confirmation dialog.

## 3. Key Features & Workflows

### A. Pipeline Generation (LLM)
- User types "Detect coins" → LLM generates JSON pipeline → Loaded into Editor.
- **[New]** Auto-searches Knowledge Base first (similarity > 0.85) before using LLM.

### B. Auto-Tuning (Optimization)
- **Input**: Source Image + Target Mask (Binary).
- **Process**: 
    -   GA tries to mutate parameters (Hill Climbing).
    -   Structural Mutation tries to Add/Remove nodes.
    -   **Safety**: Resizes are locked to aspect ratio to prevent distortion.

### C. Knowledge Retrieval & Management
- **Smart Suggest**: CLIP extracts visual features → FAISS search.
- **Save to Folder**: Now exports JSON + prompts to save to Knowledge Base.

### D. Comparison Feature **[New]**
- "📌 Save as Reference" button to store current result.
- Side-by-side comparison: Current Result vs Reference.
- Independent zoom controls for both images.

### E. Auto-Execution **[New]**
- Pipeline auto-runs when:
  - Adding/Deleting/Moving nodes
  - Enabling/Disabling nodes
  - Adjusting parameters
- Shows "✅ Pipeline Updated" toast notification.

### F. Node Reset **[New]**
- Each node has "↺ Reset" button to restore default parameters.
- Fetches original defaults from `tech_lib.json`.

## 4. Tech Theme UI **[Major Update]**

### Visual Style
- **Theme**: Cyberpunk / Tech / Sci-Fi
- **Colors**: 
  - Primary: Neon Cyan (#00ffff)
  - Secondary: Tech Blue (#00ccff, #0080ff)
  - Background: Deep dark (rgba(10, 10, 15))
  - Text: White with glow effects
- **Effects**:
  - Neon glows on all interactive elements
  - Animated scanline across page
  - Mouse trail with cyan dots
  - Click ripple effects
  - Hover lift and glow animations

### Typography
- **Fonts**: Inter + JetBrains Mono
- **Headers**: Neon cyan with text-shadow glow
- **Body**: White with high contrast

### Components
- **Buttons**: Gradient tech blue with glow
- **Inputs**: Dark background with cyan borders
- **Cards**: Glassmorphism (blur + transparency)
- **File Uploader**: Dark background with dashed cyan border
- **Dropdowns**: Dark menu with cyan text
- **Tooltips**: Dark background with cyan border

### Files Modified
- `components/style.py`: Complete rewrite (500+ lines of CSS)
  - Global text visibility fixes
  - Component-specific styling
  - Mouse effects (JavaScript injection)
  - Animation keyframes

## 5. New Algorithms Added **[New 5 Algorithms]**

Added to `tech_lib.json` and implemented in `processor.py`:

1. **Perspective Transform** (`cv_perspective_v1`)
   - Category: Geometric
   - Function: `cv2.warpPerspective`
   - Use: Correct skewed camera angles, restore ellipses to circles

2. **Watershed Segmentation** (`cv_watershed_v1`)
   - Category: Segmentation
   - Function: `cv2.watershed`
   - Use: Separate overlapping coins, ultimate solution for boundary blur

3. **Distance Transform** (`cv_dist_trans_v1`)
   - Category: Analysis
   - Function: `cv2.distanceTransform`
   - Use: Find object centers, assist Watershed separation

4. **Hu Moments** (`cv_hu_moments_v1`)
   - Category: Feature
   - Function: `cv2.HuMoments`
   - Use: Low-power shape descriptor, Hu[0] ≈ 0.16 indicates circle

5. **Fast CLAHE** (`cv_fast_clahe_v2`)
   - Category: Enhancement
   - Function: `cv2.createCLAHE`
   - Use: 40% faster than traditional CLAHE, optimized for coin reflection

Implementation locations:
- `transform.py`: `op_perspective_transform`
- `segmentation.py`: `op_watershed` (new file)
- `detect.py`: `op_distance_transform`
- `feature.py`: `op_hu_moments`
- `filter.py`: `op_fast_clahe`

## 6. Bug Fixes Applied Today

### Critical Fixes
1. **[FIXED]** Algorithm name mismatch: `cv2.GaussianBlur` vs `gaussian_blur`
   - Solution: `_normalize_func_name()` in `processor.py`
   
2. **[FIXED]** Skip nodes breaking pipeline execution
   - Solution: Track `last_valid_output` and pass through disabled nodes

3. **[FIXED]** Knowledge Base loading not copying parameters
   - Solution: Use `copy.deepcopy()` when loading cases

4. **[FIXED]** Text visibility in dark theme
   - Multiple iterations to fix white-on-white issues
   - Final solution: Global `color: #ffffff !important` with specific overrides

### UI Fixes
5. **[FIXED]** File uploader white background
   - Dark background with dashed cyan border

6. **[FIXED]** Help tooltips white background
   - Dark background + cyan text for all tooltip variants

7. **[FIXED]** Dropdown menus white background
   - Aggressive CSS targeting all baseweb menu components

8. **[FIXED]** LLM Settings and Model Name dropdowns
   - Forced dark backgrounds on all select components

## 7. Maintenance & Known Issues

### File Structure
- `app/`: Core application logic
- `components/`: UI sub-modules (style.py is critical for theme)
- `tech_lib.json`: **CRITICAL**. Defines all available nodes.
- `knowledge_db.json`: Knowledge base metadata + embeddings.

### Common Pitfalls
1.  **Streamlit State**: `st.session_state` is volatile. Always ensure state sync.
2.  **OpenCV Types**: `st.number_input` crashes with `numpy.int`. Cast to `int()`.
3.  **Resize Distortion**: Use `op_resize` in `basic.py` which handles `0` as "Auto".
4.  **D3.js Integration**: Escape Python f-strings with `{{}}` when writing JS.
5.  **CSS Specificity**: The Tech Theme uses aggressive `!important` rules. When adding new components, ensure they don't get overridden.

### UI Testing Checklist
When modifying UI, verify in dark theme:
- [ ] Text is visible (not white-on-white)
- [ ] Tooltips show dark background
- [ ] Dropdown menus are dark
- [ ] File uploader is visible
- [ ] Buttons have glow effects
- [ ] No layout breaks on mobile/narrow screens

### Roadmap (Future Agents)
- [ ] **Cloud Storage**: Move `knowledge_db.json` to Google Cloud Storage.
- [ ] **Real LLM Feedback**: Connect GPT-4o-Vision to `AutoTuner`.
- [ ] **FPGA Export**: Enhance `code_generator.py` with HLS pragmas.
- [ ] **Theme Toggle**: Light/Dark mode switch (currently locked to dark tech theme).

---
*Handed off by Sisyphus Agent - Tech Theme Edition*
*Total commits today: 22*
*Major features: Auto-execution, Comparison, 5 New Algorithms, Complete Tech Theme*
