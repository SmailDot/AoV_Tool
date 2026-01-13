import cv2
import numpy as np
import itertools

# TWD Specs
COIN_DIAMS = {
    "1": 20.0,
    "5": 22.0,
    "10": 26.0,
    "20": 26.85,
    "50": 28.0
}
# Ratios relative to 1 TWD (20mm)
# 1: 1.0
# 5: 1.1
# 10: 1.3
# 20: 1.34
# 50: 1.4

def check_ratios(radii):
    if len(radii) < 2:
        return 0, 0 # Score, Count matching
    
    # Sort radii
    radii = sorted(radii)
    
    # Try to fit to TWD models
    # Hypothesize that radii[0] is 1 TWD? 
    # Or Hypothesize that radii[0] is X TWD?
    
    best_score = 0
    
    # Let's try to find the Largest Consistent Subset (LCS) that fits a linear model
    # diameter = k * radius
    # We want to find k such that radius * k is close to one of the COIN_DIAMS
    
    # Brute force 'k' by assuming one coin is a match
    # Try assuming each detected circle is a '1 TWD' and see if others fit
    # Try assuming each detected circle is a '10 TWD' and see if others fit
    
    max_matches = 0
    best_k = 0
    
    for r_ref in radii:
        for coin_name, coin_d in COIN_DIAMS.items():
            # Assume r_ref corresponds to coin_d
            # k = coin_d / r_ref (mm per pixel * 2) -> actually just scale factor
            # predicted_diam = r * (coin_d / r_ref)
            
            k = coin_d / r_ref 
            
            matches = 0
            for r in radii:
                pred_d = r * k
                # Check if pred_d is close to ANY coin
                is_match = False
                for c_n, c_d in COIN_DIAMS.items():
                    if abs(pred_d - c_d) < 1.0: # 1mm tolerance
                        is_match = True
                        break
                if is_match:
                    matches += 1
            
            if matches > max_matches:
                max_matches = matches
                best_k = k
                
    return max_matches, best_k

def run_tuning(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    
    print(f"Image: {image_path} {img.shape}")
    
    # Parameter Sweep
    # param1: edge threshold (higher = fewer edges)
    # param2: accumulator threshold (higher = fewer circles, more perfect)
    # minDist: distance between centers
    
    results = []
    
    for p1 in [50, 70, 100]:
        for p2 in [30, 40, 50, 60, 70]:
            for min_dist in [30, 50, 70]:
                 circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, 
                    dp=1.2, 
                    minDist=min_dist,
                    param1=p1, 
                    param2=p2, 
                    minRadius=15, # Adjusted for 302px width (10mm ~ ?)
                    maxRadius=60  # If 50TWD is huge, maybe up to 60? In 300px width, 28mm is likely < 60px radius
                )
                 
                 if circles is not None:
                     count = circles.shape[1]
                     radii = circles[0, :, 2]
                     matches, k = check_ratios(radii)
                     results.append((p1, p2, min_dist, count, matches, k, radii))
                 else:
                     results.append((p1, p2, min_dist, 0, 0, 0, []))

    # Sort by matches (desc), then count (asc) -> preferring cleaner fit
    results.sort(key=lambda x: (x[4], -x[3]), reverse=True)
    
    print("Top 5 Configs:")
    for res in results[:5]:
        p1, p2, md, count, matches, k, radii = res
        print(f"P1={p1} P2={p2} MinDist={md} -> Found {count} circles, {matches} fit TWD model. Radii: {np.sort(radii).astype(int)}")

if __name__ == "__main__":
    run_tuning("test1.jpg")
