PCB Instance Segmentation — Model V5

Custom TensorFlow/Keras instance-segmentation model for detecting and separating geometric PCB features in 512 × 512 images.

The model combines semantic classification, object-center prediction, pixel-to-center offsets, and boundary prediction. These outputs are decoded into separate object instances, allowing multiple objects of the same class to receive different instance IDs.

Classes

ID

Class

0

Background

1

Rectangle

2

Rectangle_concave

3

circle

4

circle_full

Model outputs

Output head

Purpose

Semantic

Assigns one of the five classes to every pixel

Center

Predicts the center of each foreground object

Offset

Points foreground pixels toward their object center

Boundary

Helps separate touching or closely spaced objects

The instance decoder applies center non-maximum suppression, assigns foreground pixels to detected centers, and removes very small predicted instances.

Validation results

The reported results were produced from the best checkpoint on 864 validation images. The checkpoint was selected using validation foreground mean IoU.

Overall results

Metric

Result

Validation images

864

Image size

512 × 512

Pixel accuracy

97.05%

Mean IoU, including background

86.65%

Foreground mean IoU

84.15%

Instance precision at IoU 0.50

84.82%

Instance recall at IoU 0.50

95.02%

Instance F1 at IoU 0.50

89.63%

True positives

34,898

False positives

6,244

False negatives

1,830

Semantic segmentation by class

Class

IoU

Precision

Recall

Dice/F1

Background

96.64%

98.00%

98.59%

98.29%

Rectangle

84.33%

94.82%

88.40%

91.50%

Rectangle_concave

77.04%

79.90%

95.55%

87.03%

circle

89.48%

94.62%

94.28%

94.45%

circle_full

85.75%

94.22%

90.50%

92.33%

Instance segmentation by class

Class

Precision

Recall

F1

Rectangle

75.96%

94.31%

84.15%

Rectangle_concave

54.03%

95.38%

68.98%

circle

86.13%

96.73%

91.13%

circle_full

87.56%

94.74%

91.01%

The main remaining weakness is Rectangle_concave instance precision. Its high recall and lower precision indicate that the decoder finds most real objects but also produces too many false-positive concave rectangles.

Training summary

Maximum configured training length: 400 epochs

Best checkpoint: epoch 297

Best validation foreground mean IoU: 0.84150

Early stopping: epoch 357

Training uses a learning-rate scheduler and mixed-precision-compatible TensorFlow execution.

Online augmentation includes rotations and mild photometric/geometric variations.

The best checkpoint is saved independently from the final training epoch.





Repository contents

The important generated files and directories are:

model-v5-instance/
├── model_v5.py
├── best_model_v5_instance.keras
├── final_best_model_v5_instance.keras
├── training_log.csv
├── training_target_preview.png
├── online_augmentation_catalog.png
├── validation_previews/
├── performance/
│   ├── training_performance_live.png
│   ├── training_performance_final.png
│   ├── training_performance_summary.json
│   └── val_final_evaluation/
└── tensorboard/

best_model_v5_instance.keras is the recommended checkpoint for evaluation and inference. final_best_model_v5_instance.keras is a final copy of the same selected best model.

Clone the repository

Model files are stored with Git Large File Storage. Install Git LFS before downloading the checkpoints.

git lfs install
git clone https://github.com/Jenit88/model-v5-instance.git
cd model-v5-instance
git lfs pull

Environment

The trained checkpoint was produced with TensorFlow 2.21 in a GPU-enabled Linux/WSL environment. A CUDA-capable GPU is recommended for training.

Core Python dependencies include:

python -m pip install tensorflow numpy opencv-python matplotlib scipy pillow

If an existing project environment is available, use its pinned package versions instead of replacing them with the newest releases.

Configure the project

Before running the script, open model_v5.py and verify:

Training and validation dataset paths.

Model output directory.

Image size and class mapping.

Fresh-start or resume-training settings.

Requested run mode.

The original project uses absolute WSL paths. Replace them when running the project on another computer.

For resuming the existing training workflow, the configuration used was:

START_FRESH = False
RESET_TRAINING = False
RESUME_TRAINING = True

Only enable resume mode when the expected backup/checkpoint files are available.

Run training or evaluation

Activate the TensorFlow environment and run:

python model_v5.py

The script writes checkpoints, previews, performance summaries, and final semantic/instance evaluation reports into the configured model output directory.

TensorBoard

To view the saved TensorBoard logs:

tensorboard --logdir tensorboard --port 6006

Then open http://localhost:6006 in a browser. When running through WSL, open the address using the Windows browser.

Instance decoder settings

The final validation report used:

Setting

Value

Semantic confidence

0.30

Center confidence

0.12

Center NMS radius

4 pixels

Maximum assignment distance

61 pixels

Minimum instance area

13 pixels

These thresholds affect precision, recall, and instance separation. Re-evaluate the complete validation or test set after changing them.

Evaluation notes

The semantic and instance results come from the validation split, not an independent unseen test set.

Instance precision, recall, and F1 use an IoU matching threshold of 0.50.

These values are not COCO mAP50 or mAP50–95.

Inference must use the same image normalization, resizing/letterboxing, class order, and decoder configuration used during validation.

Threshold tuning should be performed on validation data only; reserve a separate test set for the final unbiased result.

Output reports

The final evaluation produces:

performance/val_final_evaluation/val_semantic_evaluation.json
performance/val_final_evaluation/val_instance_evaluation.json
performance/val_final_evaluation/val_final_performance_dashboard.png

These JSON files contain the full confusion matrix, per-class semantic metrics, instance counts, and decoder thresholds used for the reported results.
