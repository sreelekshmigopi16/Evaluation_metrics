import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc,average_precision_score

# Load data
gt_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/_annotations.csv')

pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/test1_fasterRCNN_single_enhanced.csv')
#pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/test1_single_mobilenet_final.csv')

#pred_df = pd.read_csv('C:/Clover_Datasets/Clover.v1i.yolov5pytorch/test1_mobilenet_v3_large_320_fpn.csv')
# Strip any extra spaces from column names
gt_df.columns = gt_df.columns.str.strip()
pred_df.columns = pred_df.columns.str.strip()

# Function to compute IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Set IoU threshold
iou_threshold = 0.5

# Initialize lists for true positives, false positives, and scores
y_true = []
y_scores = []

# Initialize counters and dictionaries
tp_count = 0
fp_count = 0
fn_count = 0
fp_images = {}
fn_images = {}
multiple_detections = {}

# Get all unique filenames from both ground truth and predictions
all_images = set(gt_df['filename'].unique()) | set(pred_df['filename'].unique())
# Process each image
for image in all_images:
    gt = gt_df[gt_df['filename'] == image]
    preds = pred_df[pred_df['filename'] == image]
    
    gt_boxes = gt[['xmin', 'ymin', 'xmax', 'ymax']].values if not gt.empty else []
    pred_boxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values if not preds.empty else []
    scores = preds['confidence'].values if not preds.empty else []
    matched_gt = set()
    image_tp = 0
    image_fp = 0
    
    for pred_box, score in zip(pred_boxes, scores):
        y_scores.append(score)

        if len(gt_boxes) > 0:
            # Calculate IoU for each prediction with all ground truth boxes
            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
            max_iou = max(ious)
            max_iou_index = np.argmax(ious)
            
            if max_iou >= iou_threshold:
                # Count as true positive even if it's a duplicate detection
                y_true.append(1)
                tp_count += 1
                image_tp += 1
                matched_gt.add(max_iou_index)
            else:
                # No match found, count as false positive
                y_true.append(0)
                fp_count += 1
                image_fp += 1
        else:
            # This is a false positive (detection in an image with no ground truth)
            y_true.append(0)
            fp_count += 1
            image_fp += 1
    
    # Count false negatives
    image_fn = len(gt_boxes) - len(matched_gt)
    fn_count += image_fn
    
    if image_fp > 0:
        fp_images[image] = image_fp
    if image_fn > 0:
        fn_images[image] = image_fn
    if len(pred_boxes) > len(gt_boxes):
        multiple_detections[image] = len(pred_boxes) - len(gt_boxes)

# Convert to numpy arrays
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Calculate precision-recall values
precision, recall, _ = precision_recall_curve(y_true, y_scores)
# Calculate ROC curve values
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
# Calculate metrics
total_gt = tp_count + fn_count
total_pred = tp_count + fp_count
overall_precision = tp_count / total_pred if total_pred > 0 else 0
overall_recall = tp_count / total_gt if total_gt > 0 else 0
f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
ap = average_precision_score(y_true, y_scores)

# Print results
print(f'True Positives: {tp_count}')
print(f'False Positives: {fp_count}')
print(f'False Negatives: {fn_count}')
print(f'Overall Precision: {overall_precision:.4f}')
print(f'Average Precision (AP): {ap:.4f}')
print(f'Recall: {overall_recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')


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


