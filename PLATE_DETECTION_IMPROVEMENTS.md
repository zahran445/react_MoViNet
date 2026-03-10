# License Plate Recognition Improvements

## Issues Identified

1. **Violation 0001**: Vehicle violation with no detectable plate in the video
   - The license plate is not visible/readable in any frame of the video clip
   - This is a video quality/angle issue, not a code issue

2. **Original System**: Only checked a single snapshot frame for plate detection
   - If the plate wasn't visible at that exact moment, detection failed

3. **Confidence Threshold**: Was too high (0.1) for some challenging detections

## Improvements Implemented

### 1. Multi-Frame Plate Detection

- **Before**: Only checked the peak violation frame
- **After**: Samples multiple frames (±0.5s around peak) to find the best plate view
- **Benefit**: Increases chances of finding a clear plate view

### 2. Lower Confidence Threshold

- **Before**: 0.1 (10%) minimum confidence
- **After**: 0.05 (5%) minimum confidence
- **Benefit**: Can detect plates in more challenging conditions

### 3. Enhanced Detection Logic

- The system now tries multiple frames before giving up
- Falls back to single-frame detection if multi-frame fails
- Better handling of different image orientations

## Results

### Current Status

- **Total Vehicle Violations**: 2
- **With Plates Detected**: 1 (violation_0009)
- **Without Plates**: 1 (violation_0001 - plate not visible in video)

### Violation 0009 (Success)

- Confidence: 99.8%
- Plate Text: 'STR242L3ZHJ'
- Status: ✓ Working correctly

### Violation 0001 (Limitation)

- Confidence: 67.3% (vehicle detected)
- Plate: Not visible in any frame of the video
- Status: ⚠️ Hardware/video quality limitation

## Recommendations

### Short-term (Code)

1. ✅ Multi-frame sampling (IMPLEMENTED)
2. ✅ Lower confidence threshold (IMPLEMENTED)
3. Consider: Increase video clip duration from 6s to 10s for more frames to sample
4. Consider: Add preprocessing (sharpening, contrast enhancement) before OCR

### Medium-term (Model)

1. Retrain YOLOv8 model with more diverse plate examples
2. Train on different angles and lighting conditions
3. Add data augmentation for portrait/rotated images

### Long-term (Hardware)

1. Use higher resolution cameras (currently 1080p-2160p)
2. Adjust camera angles to better capture license plates
3. Improve lighting conditions
4. Add redundant cameras for different angles

## Testing

The improved system can be tested by:

1. Processing new vehicle violation videos
2. Running: `python test_multiframe_detection.py`
3. Running: `python audit_violations.py`

## Conclusion

The system improvements make plate detection more robust, but cannot overcome
fundamental limitations like plates not being visible in the video. For violation_0001,
the plate is simply not visible/readable in the captured footage.
