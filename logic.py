import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import base64
from datetime import datetime
from collections import defaultdict

# Meat freshness classes and colors (BGR for OpenCV)
MEAT_FRESHNESS_CLASSES = {
    'fresh_meat': {'color': (0, 255, 0), 'label': 'Fresh Meat', 'emoji': '✅'},
    'medium_fresh': {'color': (0, 255, 255), 'label': 'Medium Fresh', 'emoji': '⚠️'},
    'aging_meat': {'color': (0, 165, 255), 'label': 'Aging Meat', 'emoji': '🟡'},
    'spoiled_meat': {'color': (0, 0, 255), 'label': 'Spoiled Meat', 'emoji': '❌'}
}

def analyze_color_features(cv_image, bbox):
    """Extract and analyze color features for meat freshness assessment"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        # Clip coordinates to image bounds
        h, w = cv_image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        # Extract ROI
        roi = cv_image[y1:y2, x1:x2]
        if roi.size == 0:
            return {'brightness': 0.5, 'redness': 0.5, 'saturation': 0.5}
        
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Analyze hue (color tone - red indicates freshness)
        hue = hsv_roi[:,:,0]
        saturation = hsv_roi[:,:,1]
        value = hsv_roi[:,:,2]
        
        # Red meat should have red hue (0-10 or 170-180 in OpenCV 0-180 scale)
        red_hue_mask = ((hue < 10) | (hue > 170)).astype(float)
        redness_score = np.mean(red_hue_mask)
        
        # Saturation indicates freshness (high = fresh)
        saturation_score = np.mean(saturation) / 255.0
        
        # Brightness/Value
        brightness_score = np.mean(value) / 255.0
        
        return {
            'redness': redness_score,
            'saturation': saturation_score,
            'brightness': brightness_score
        }
    except Exception as e:
        print(f"Error in color analysis: {e}")
        return {'brightness': 0.5, 'redness': 0.5, 'saturation': 0.5}

def get_meat_freshness_class(confidence, bbox, image_shape, cv_image=None):
    """Advanced meat freshness classification using multiple features"""
    x1, y1, x2, y2 = bbox
    roi_width = x2 - x1
    roi_height = y2 - y1
    
    # Size-based confidence adjustment
    size_factor = min((roi_width * roi_height) / (image_shape[0] * image_shape[1]), 1.0)
    adjusted_confidence = confidence * (0.7 + 0.3 * size_factor)
    
    # Color-based analysis
    color_boost = 0.0
    if cv_image is not None:
        colors = analyze_color_features(cv_image, bbox)
        # Fresh meat has: high redness + high saturation + moderate brightness
        freshness_indicator = (colors['redness'] * 0.4 + 
                             colors['saturation'] * 0.4 + 
                             (1 - abs(colors['brightness'] - 0.6)) * 0.2)
        color_boost = freshness_indicator * 0.2
    
    final_score = adjusted_confidence + color_boost
    
    if final_score > 0.82:
        return 'fresh_meat'
    elif final_score > 0.68:
        return 'medium_fresh'
    elif final_score > 0.48:
        return 'aging_meat'
    else:
        return 'spoiled_meat'

def generate_meat_freshness_report(detections, model_name="YOLO Custom"):
    """Generate detailed meat freshness analysis report"""
    if not detections:
        return "No meat products detected in image."
    
    freshness_counts = defaultdict(int)
    confidence_scores = [d['confidence'] for d in detections]
    
    for detection in detections:
        freshness_counts[detection['class']] += 1
    
    total_items = len(detections)
    avg_confidence = np.mean(confidence_scores)
    max_confidence = max(confidence_scores)
    
    report = f"🍖 **Meat Freshness AI Analysis Report**\n"
    report += f"Generated: {datetime.now().strftime('%d %B %Y at %I:%M %p')}\n"
    report += f"Model: {model_name}\n"
    report += "=" * 40 + "\n\n"
    
    report += "📊 **Detection Performance:**\n"
    report += f"  🔍 Total Items: {total_items}\n"
    report += f"  📈 Avg Confidence: {avg_confidence:.1%}\n"
    report += f"  ⭐ Max Confidence: {max_confidence:.1%}\n\n"
    
    report += "🥩 **Freshness Classification:**\n"
    for cls, count in freshness_counts.items():
        percentage = (count / total_items) * 100
        info = MEAT_FRESHNESS_CLASSES[cls]
        report += f"  {info['emoji']} {info['label']}: {count} ({percentage:.1f}%)\n"
    
    report += f"\n💡 **AI Recommendations:**\n"
    spoiled = freshness_counts.get('spoiled_meat', 0)
    aging = freshness_counts.get('aging_meat', 0)
    
    if spoiled > 0:
        report += "  🚨 **CRITICAL**: Spoiled items detected! Dispose immediately.\n"
    if aging > 0:
        report += "  ⏰ **ACTION**: Aging meat found. Use or freeze NOW.\n"
    if spoiled == 0 and aging == 0:
        report += "  ✅ **EXCELLENT**: All items appear fresh and safe.\n"
        
    return report
