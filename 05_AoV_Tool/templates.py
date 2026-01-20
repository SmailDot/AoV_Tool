from typing import List, Dict, Any

def get_default_templates() -> Dict[str, List[Dict[str, Any]]]:
    """
    回傳預設的 Pipeline 模板
    """
    return {
        "硬幣偵測 (Coin Detection)": [
            {"id": "node_0", "function": "median_blur", "name": "降噪", "parameters": {"ksize": {"default": 5}}},
            {"id": "node_1", "function": "canny_edge", "name": "邊緣偵測", "parameters": {"threshold1": {"default": 100}, "threshold2": {"default": 200}}},
            {"id": "node_2", "function": "hough_circles", "name": "圓形偵測", "parameters": {"minDist": {"default": 50}, "param1": {"default": 50}, "param2": {"default": 30}, "minRadius": {"default": 20}, "maxRadius": {"default": 100}}}
        ],
        "人臉偵測 (Face Detection)": [
            {"id": "node_0", "function": "cascade_classifier", "name": "Haar 人臉偵測", "parameters": {"model_type": {"default": "haarcascade_frontalface_default.xml"}, "scaleFactor": {"default": 1.1}, "minNeighbors": {"default": 5}}}
        ],
        "行人偵測 (Pedestrian Detection)": [
            {"id": "node_0", "function": "hog_descriptor", "name": "HOG 行人偵測", "parameters": {"scale": {"default": 1.05}}}
        ],
        "車輛偵測 (Vehicle - Moving)": [
            {"id": "node_0", "function": "gaussian_blur", "name": "模糊", "parameters": {"ksize": {"default": [5, 5]}}},
            {"id": "node_1", "function": "background_subtractor", "name": "背景分割", "parameters": {"history": {"default": 500}, "varThreshold": {"default": 16}}},
            {"id": "node_2", "function": "morph_open", "name": "去除雜訊", "parameters": {"kernel_size": {"default": 3}}},
            {"id": "node_3", "function": "find_contours", "name": "標記物件", "parameters": {"min_area": {"default": 500}}}
        ],
        "車牌定位 (License Plate)": [
            {"id": "node_0", "function": "bgr2gray", "name": "轉灰階", "parameters": {}},
            {"id": "node_1", "function": "sobel", "name": "垂直邊緣 (Sobel X)", "parameters": {"dx": {"default": 1}, "dy": {"default": 0}}},
            {"id": "node_2", "function": "threshold_binary", "name": "二值化", "parameters": {"thresh": {"default": 0}, "type": {"default": 8}}}, # OTSU? No, explicit usually.
            {"id": "node_3", "function": "morph_close", "name": "閉運算 (連接)", "parameters": {"kernel_size": {"default": 17}, "iterations": {"default": 1}}}, # Wide kernel for plates
            {"id": "node_4", "function": "find_contours", "name": "輪廓標記", "parameters": {"min_area": {"default": 1000}}}
        ],
        "物體追蹤 (Object Tracking - Color)": [
            {"id": "node_0", "function": "bgr2hsv", "name": "轉 HSV", "parameters": {}},
            {"id": "node_1", "function": "in_range", "name": "顏色過濾", "parameters": {"lower_b": {"default": 0}, "upper_b": {"default": 180}, "lower_g": {"default": 0}, "upper_g": {"default": 255}, "lower_r": {"default": 0}, "upper_r": {"default": 255}}},
            {"id": "node_2", "function": "morph_open", "name": "去除雜訊", "parameters": {}},
            {"id": "node_3", "function": "find_contours", "name": "標記物體", "parameters": {}}
        ]
    }
