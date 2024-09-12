import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

# Load data
#pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/pboxes_multi_yolo_final.csv')
#pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/test_attention.csv')
pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/test_final_multi_Rcnn.csv')
#pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/test1_multi_mobilenet_final (1).csv')

gt_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/groundtruth.csv')

# Strip any extra spaces from column names
gt_df.columns = gt_df.columns.str.strip()
pred_df.columns = pred_df.columns.str.strip()

# Print column names to verify
print("Ground truth columns:", gt_df.columns)
print("Predicted columns:", pred_df.columns)

# Helper function to compute IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Set IoU threshold
iou_threshold = 0.4

# Initialize lists to store true positives, false positives, and confidence scores
y_true = []
y_scores = []

# Initialize counters
tp_count = 0
fp_count = 0
fn_count = 0

# Process each image
for image_name, preds in pred_df.groupby('filename'):
    # Ground truth for this image
    gt = gt_df[gt_df['filename'] == image_name]
    
    gt_boxes = gt[['xmin', 'ymin', 'xmax', 'ymax']].values if not gt.empty else []
    pred_boxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values
    confidences = preds['confidence'].values
    
    matched_gt = set()
    
    for pred_box, confidence in zip(pred_boxes, confidences):
        y_scores.append(confidence)
        
        if len(gt_boxes) > 0:
            # Calculate IoU for each prediction with all ground truth boxes
            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
            max_iou = max(ious)
            max_iou_index = np.argmax(ious)
            
            if max_iou >= iou_threshold:
                y_true.append(1)
                tp_count += 1
                matched_gt.add(max_iou_index)
            else:
                y_true.append(0)
                fp_count += 1
        else:
            y_true.append(0)
            fp_count += 1
    
    # Count false negatives
    fn_count += len(gt_boxes) - len(matched_gt)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Calculate overall precision and recall
precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

# Calculate F1 Score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f'True Positives: {tp_count}')
print(f'False Positives: {fp_count}')
print(f'False Negatives: {fn_count}')
print(f'Overall Precision: {precision:.4f}')
print(f'Overall Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Calculate precision-recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Calculate Average Precision (AP)
ap = average_precision_score(y_true, y_scores)

# Plot Precision-Recall curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall_curve, precision_curve, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.text(0.05, 0.05, f'AP: {ap:.4f}', transform=plt.gca().transAxes)
plt.grid(True)

# Plot ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()

print(f'Average Precision (AP): {ap:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')