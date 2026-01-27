"""
Code Generator for NKUST AoV Tool
負責人：System_Architect (Code_Synthesizer)

職責：
1. 將 Pipeline JSON 轉換為可執行的 Python 腳本 (OpenCV)
2. 將 Pipeline JSON 轉換為 HLS C++ 概念代碼 (FPGA High-Level Synthesis)
"""

from typing import List, Dict, Any
import json
from datetime import datetime

class CodeGenerator:
    """
    代碼生成器
    """
    
    @staticmethod
    def generate_python_script(pipeline: List[Dict[str, Any]]) -> str:
        """
        生成 Python (OpenCV) 腳本
        """
        script = []
        
        # Header
        script.append(f"# NKUST AoV Tool - Generated Python Script")
        script.append(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script.append(f"# Total Nodes: {len(pipeline)}")
        script.append("")
        script.append("import cv2")
        script.append("import numpy as np")
        script.append("import sys")
        script.append("")
        
        # Main Function
        script.append("def main():")
        script.append("    # Load Image")
        script.append("    if len(sys.argv) < 2:")
        script.append("        print('Usage: python pipeline.py <image_path>')")
        script.append("        # Creating a dummy image for demonstration")
        script.append("        img = np.zeros((480, 640, 3), dtype=np.uint8)")
        script.append("    else:")
        script.append("        img = cv2.imread(sys.argv[1])")
        script.append("        if img is None:")
        script.append("            print('Error: Could not load image')")
        script.append("            return")
        script.append("")
        script.append("    # Pipeline Execution")
        script.append("    current_img = img.copy()")
        script.append("")
        
        # Nodes
        for idx, node in enumerate(pipeline):
            if not node.get('_enabled', True):
                script.append(f"    # Node {idx}: {node['name']} (Disabled)")
                continue
                
            script.append(f"    # Node {idx}: {node['name']}")
            func_name = node.get('function', 'unknown')
            params = node.get('parameters', {})
            
            # Param parsing
            args_str = "current_img"
            
            # Special handling for common functions based on tech_lib.json schema
            if func_name == "GaussianBlur":
                ksize = params.get('ksize', {}).get('default', [5, 5])
                sigmaX = params.get('sigmaX', {}).get('default', 0)
                sigmaY = params.get('sigmaY', {}).get('default', 0)
                script.append(f"    current_img = cv2.GaussianBlur(current_img, {tuple(ksize)}, {sigmaX}, {sigmaY})")
                
            elif func_name == "Canny":
                # Canny requires gray
                script.append("    if len(current_img.shape) == 3:")
                script.append("        gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)")
                script.append("    else:")
                script.append("        gray = current_img")
                
                t1 = params.get('threshold1', {}).get('default', 50)
                t2 = params.get('threshold2', {}).get('default', 150)
                ap = params.get('apertureSize', {}).get('default', 3)
                script.append(f"    edges = cv2.Canny(gray, {t1}, {t2}, apertureSize={ap})")
                script.append("    current_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Convert back to BGR for display")
                
            elif func_name == "Dilate":
                k = params.get('kernel', {}).get('default', [3, 3]) # Assuming simplified representation
                iter_count = params.get('iterations', {}).get('default', 1)
                script.append(f"    kernel = np.ones((3, 3), np.uint8) # Simplification")
                script.append(f"    current_img = cv2.dilate(current_img, kernel, iterations={iter_count})")
                
            elif func_name == "Erode":
                iter_count = params.get('iterations', {}).get('default', 1)
                script.append(f"    kernel = np.ones((3, 3), np.uint8)")
                script.append(f"    current_img = cv2.erode(current_img, kernel, iterations={iter_count})")
                
            elif func_name == "HoughCircles":
                script.append("    if len(current_img.shape) == 3:")
                script.append("        gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)")
                script.append("    else:")
                script.append("        gray = current_img")
                
                dp = params.get('dp', {}).get('default', 1.2)
                minDist = params.get('minDist', {}).get('default', 30)
                p1 = params.get('param1', {}).get('default', 50)
                p2 = params.get('param2', {}).get('default', 30)
                minR = params.get('minRadius', {}).get('default', 0)
                maxR = params.get('maxRadius', {}).get('default', 0)
                
                script.append(f"    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, {dp}, {minDist}, param1={p1}, param2={p2}, minRadius={minR}, maxRadius={maxR})")
                script.append("    # Draw circles")
                script.append("    if circles is not None:")
                script.append("        circles = np.uint16(np.around(circles))")
                script.append("        for i in circles[0, :]:")
                script.append("            cv2.circle(current_img, (i[0], i[1]), i[2], (0, 255, 0), 2)")
                script.append("            cv2.circle(current_img, (i[0], i[1]), 2, (0, 0, 255), 3)")
                
            elif func_name == "Threshold":
                script.append("    if len(current_img.shape) == 3:")
                script.append("        gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)")
                script.append("    else:")
                script.append("        gray = current_img")
                
                thresh = params.get('thresh', {}).get('default', 127)
                maxval = params.get('maxval', {}).get('default', 255)
                script.append(f"    _, binary = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_BINARY)")
                script.append("    current_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)")

            elif func_name == "MorphologicalOpen":
                script.append("    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))")
                script.append("    current_img = cv2.morphologyEx(current_img, cv2.MORPH_OPEN, kernel)")
                
            elif func_name == "CLAHE":
                script.append("    lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)")
                script.append("    l, a, b = cv2.split(lab)")
                clip = params.get('clipLimit', {}).get('default', 2.0)
                tile = params.get('tileGridSize', {}).get('default', [8, 8])
                script.append(f"    clahe = cv2.createCLAHE(clipLimit={clip}, tileGridSize={tuple(tile)})")
                script.append("    l_clahe = clahe.apply(l)")
                script.append("    lab_clahe = cv2.merge((l_clahe, a, b))")
                script.append("    current_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)")
                
            else:
                script.append(f"    # TODO: Implement {func_name} manually")
                script.append(f"    # Params: {params}")
            
            script.append("")
        
        script.append("    # Display Result")
        script.append("    cv2.imshow('Result', current_img)")
        script.append("    cv2.waitKey(0)")
        script.append("    cv2.destroyAllWindows()")
        script.append("")
        
        script.append("if __name__ == '__main__':")
        script.append("    main()")
        
        return "\n".join(script)

    @staticmethod
    def generate_vhdl(pipeline: List[Dict[str, Any]]) -> str:
        """
        生成 Vivado VHDL 概念代碼
        """
        script = []
        script.append("-- NKUST AoV Tool - Generated Vivado VHDL Concept")
        script.append(f"-- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script.append("")
        script.append("library IEEE;")
        script.append("use IEEE.STD_LOGIC_1164.ALL;")
        script.append("use IEEE.NUMERIC_STD.ALL;")
        script.append("")
        script.append("entity ImagePipeline is")
        script.append("    Port (")
        script.append("        aclk        : in STD_LOGIC;")
        script.append("        aresetn     : in STD_LOGIC;")
        script.append("        s_axis_tdata  : in STD_LOGIC_VECTOR(23 downto 0);")
        script.append("        s_axis_tvalid : in STD_LOGIC;")
        script.append("        s_axis_tready : out STD_LOGIC;")
        script.append("        m_axis_tdata  : out STD_LOGIC_VECTOR(23 downto 0);")
        script.append("        m_axis_tvalid : out STD_LOGIC;")
        script.append("        m_axis_tready : in STD_LOGIC")
        script.append("    );")
        script.append("end ImagePipeline;")
        script.append("")
        script.append("architecture Behavioral of ImagePipeline is")
        script.append("    -- Signal definitions for pipeline stages")
        for idx in range(len(pipeline) + 1):
             script.append(f"    signal data_stage_{idx} : STD_LOGIC_VECTOR(23 downto 0);")
             script.append(f"    signal valid_stage_{idx} : STD_LOGIC;")
        script.append("begin")
        script.append("")
        script.append("    -- Dataflow Logic (Simplified)")
        script.append("    s_axis_tready <= m_axis_tready; -- Pass-through flow control")
        script.append("")
        
        for idx, node in enumerate(pipeline):
            if not node.get('_enabled', True):
                continue
            func_name = node.get('function', 'unknown')
            script.append(f"    -- Node {idx}: {node['name']} ({func_name})")
            script.append(f"    process(aclk)")
            script.append(f"    begin")
            script.append(f"        if rising_edge(aclk) then")
            script.append(f"            if aresetn = '0' then")
            script.append(f"                valid_stage_{idx+1} <= '0';")
            script.append(f"            else")
            script.append(f"                -- Placeholder logic for {func_name}")
            script.append(f"                valid_stage_{idx+1} <= valid_stage_{idx};")
            script.append(f"                data_stage_{idx+1} <= data_stage_{idx}; -- Bypass")
            script.append(f"            end if;")
            script.append(f"        end if;")
            script.append(f"    end process;")
            script.append("")

        script.append("    -- Output assignment")
        script.append(f"    m_axis_tdata <= data_stage_{len(pipeline)};")
        script.append(f"    m_axis_tvalid <= valid_stage_{len(pipeline)};")
        script.append("")
        script.append("end Behavioral;")
        
        return "\n".join(script)

    @staticmethod
    def generate_verilog(pipeline: List[Dict[str, Any]]) -> str:
        """
        生成 Vivado Verilog 概念代碼
        """
        script = []
        script.append(f"// NKUST AoV Tool - Generated Vivado Verilog Concept")
        script.append(f"// Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script.append("")
        script.append("module ImagePipeline (")
        script.append("    input wire aclk,")
        script.append("    input wire aresetn,")
        script.append("    input wire [23:0] s_axis_tdata,")
        script.append("    input wire s_axis_tvalid,")
        script.append("    output wire s_axis_tready,")
        script.append("    output wire [23:0] m_axis_tdata,")
        script.append("    output wire m_axis_tvalid,")
        script.append("    input wire m_axis_tready")
        script.append(");")
        script.append("")
        
        for idx in range(len(pipeline) + 1):
             script.append(f"    reg [23:0] data_stage_{idx};")
             script.append(f"    reg valid_stage_{idx};")
             
        script.append("")
        script.append("    assign s_axis_tready = m_axis_tready;")
        script.append("    assign m_axis_tdata = data_stage_" + str(len(pipeline)) + ";")
        script.append("    assign m_axis_tvalid = valid_stage_" + str(len(pipeline)) + ";")
        script.append("")
        
        script.append("    always @(posedge aclk or negedge aresetn) begin")
        script.append("        if (!aresetn) begin")
        for idx in range(len(pipeline) + 1):
             script.append(f"            valid_stage_{idx} <= 1'b0;")
             script.append(f"            data_stage_{idx} <= 24'd0;")
        script.append("        end else begin")
        script.append("            // Input Stage")
        script.append("            data_stage_0 <= s_axis_tdata;")
        script.append("            valid_stage_0 <= s_axis_tvalid;")
        script.append("")
        
        for idx, node in enumerate(pipeline):
            if not node.get('_enabled', True):
                continue
            func_name = node.get('function', 'unknown')
            script.append(f"            // Node {idx}: {node['name']} ({func_name})")
            script.append(f"            data_stage_{idx+1} <= data_stage_{idx}; // Bypass logic")
            script.append(f"            valid_stage_{idx+1} <= valid_stage_{idx};")
            script.append("")
            
        script.append("        end")
        script.append("    end")
        script.append("endmodule")
        
        return "\n".join(script)

