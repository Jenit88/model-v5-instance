"""Model_v5: improved scratch-trained PCB instance segmentation.

This is a complete replacement for Model_v4. It preserves the useful
semantic + class-specific center + offset-vector instance representation, but
improves the parts that matter most for this PCB dataset:

* 512 x 512 training by default, preserving small ports and curved boundaries.
* Group Normalization for stable batches of two images on an 8 GB GPU.
* Stronger YOLO-inspired backbone, SPPF, FPN/PAN context, and skip decoder.
* A fourth auxiliary boundary head trained from the true instance-ID map.
  It teaches same-class touching-object boundaries as well as class edges.
* Numerically stable logit-based focal, Dice, CenterNet, Huber, and boundary
  losses with float32 output heads under mixed precision.
* AdamW, gradient accumulation when supported, EMA when supported, warm-up,
  cosine decay, crash recovery, TensorBoard, validation previews, and a best
  checkpoint selected by foreground mean IoU rather than total loss.
* Live and final performance dashboards with every head loss, foreground and
  per-class IoU, pixel accuracy, boundary F1, learning rate, generalization
  gap, confusion matrix, and instance precision/recall/F1 at IoU 0.50.
* Leakage-safe online augmentation: stored training samples stay unchanged.
  During every deterministic 95-epoch cycle, each image is presented once
  unchanged, through all 45 exact pure rotations (2, 4, ..., 90 degrees), and
  through 49 additional realistic augmentation variants. Validation and test
  data are never augmented.
* Every exact rotation uses zoom-to-fill affine warping, so no black corner
  background is introduced. Semantic masks and instance IDs are warped with
  nearest-neighbour interpolation, then center, offset, and boundary targets
  are regenerated from the transformed instances.
* Aspect-ratio-preserving letterbox prediction instead of shape distortion.
* Folder prediction and instance-level validation at IoU 0.50.

Scratch-training guarantee
--------------------------
The train path never imports an application backbone and never loads a model,
weights file, or checkpoint before ``fit``. Every trainable layer starts from
a fresh initializer. ``BackupAndRestore`` may recover an interrupted run in
the same output folder; that is continuation of this scratch run, not
pretraining.

Required arrays
---------------
Run the instance-dataset creator at the same IMG_SIZE first. The array folder
must contain:

    X_train.npy, X_val.npy
    Y_semantic_train.npy, Y_semantic_val.npy
    Y_instance_train.npy, Y_instance_val.npy
    Y_center_train.npy, Y_center_val.npy
    Y_offset_train.npy, Y_offset_val.npy

Y_offset must have three channels: normalized dx, normalized dy, valid-mask.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

IMG_SIZE = 512
NUM_CLASSES = 5

CLASS_NAMES = {
    0: "background",
    1: "Rectangle",
    2: "Rectangle_concave",
    3: "circle",
    4: "circle_full",
}

# Your YOLO polygon TXT files use class IDs 1, 2, 3, and 4. Keep this True.
# Set it to False only for standard YOLO labels numbered 0, 1, 2, and 3.
RAW_YOLO_CLASS_IDS_START_AT_ONE = True

CLASS_COLOURS_RGB = np.asarray(
    [
        [0, 0, 0],
        [230, 65, 65],
        [255, 165, 45],
        [60, 180, 90],
        [65, 135, 230],
    ],
    dtype=np.uint8,
)

# This folder is named "45000 images", but it contains the real dataset:
#   images/train, images/val, images/test
#   labels/train, labels/val, labels/test
DATASET_ROOT = Path(
    "/home/u117134c/Data_preprocessing/45000 images/Split_Data"
)
ARRAY_DIR = DATASET_ROOT / f"instance_npy_{IMG_SIZE}"
MODEL_OUTPUT_DIR = Path("/home/u117134c/Models/Model_v5_instance")

# train | preview_augmentation | predict | evaluate
RUN_MODE = "train"

# PREDICT_SOURCE may be one image or a folder. Folder mode is recursive.
PREDICT_SOURCE = Path(
    "/home/u117134c/Data_preprocessing/45000 images/Split_Data/images/test"
)
MODEL_FOR_INFERENCE = MODEL_OUTPUT_DIR / "best_model_v5_instance.keras"

# 512 x 512 with batch 2 is a suitable first attempt on the RTX PRO 1000 8 GB.
# If an out-of-memory error occurs, use BATCH_SIZE = 1 and keep accumulation.
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 400
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 60

USE_MIXED_PRECISION = True
USE_EMA = True
EMA_MOMENTUM = 0.999
REQUIRE_GPU = True
ENABLE_OP_DETERMINISM = False
SEED = 42

# Online augmentation. No augmented image or label files are written to disk.
# Every source image receives one original presentation, all 45 exact pure
# rotations, and all 49 existing realistic variants in each 95-epoch cycle.
USE_ONLINE_AUGMENTATION = True
AUGMENTATION_CYCLE_LENGTH = 95

# Modes 1..45 are the exact positive-angle sequence used by the supplied
# rotation code. These modes apply only rotation + zoom-to-fill. Modes 46..94
# retain the previous 49 realistic variants, whose rotation angle is sampled
# from the same 2-degree grid and may use either direction.
MIN_ROTATION_DEGREES = 2
MAX_ROTATION_DEGREES = 90
ROTATION_STEP_DEGREES = 2
ROTATE_BOTH_DIRECTIONS = True
EXACT_ROTATION_ANGLES = tuple(
    range(
        MIN_ROTATION_DEGREES,
        MAX_ROTATION_DEGREES + 1,
        ROTATION_STEP_DEGREES,
    )
)
EXACT_ROTATION_MODE_COUNT = len(EXACT_ROTATION_ANGLES)  # 45
REALISTIC_VARIANT_MODE_COUNT = 49
REALISTIC_VARIANT_MODE_OFFSET = EXACT_ROTATION_MODE_COUNT

# Do not allow early stopping before every image has completed at least one
# full original + exact-rotation + realistic-variant cycle.
MINIMUM_AUGMENTATION_CYCLES_BEFORE_EARLY_STOPPING = 1
EARLY_STOPPING_START_EPOCH = (
    MINIMUM_AUGMENTATION_CYCLES_BEFORE_EARLY_STOPPING
    * AUGMENTATION_CYCLE_LENGTH
)

# Zoom-to-fill removes rotation corners instead of adding a black background.
# Extra zoom provides a safe crop margin for the translation modes.
ZOOM_SAFETY_FACTOR = 1.002
EXTRA_ZOOM_RANGE = (1.06, 1.12)
MAX_TRANSLATION_FRACTION = 0.04

# Mild, production-realistic photometric ranges. Image values are in [0, 1].
BRIGHTNESS_LIMIT = 0.10
CONTRAST_LIMIT = 0.10
GAMMA_RANGE = (0.90, 1.10)
EXPOSURE_GAIN_RANGE = (0.95, 1.05)
NOISE_STD_RANGE = (0.005, 0.020)
HUE_SHIFT_DEGREES = 3.0
SATURATION_GAIN_RANGE = (0.92, 1.08)

# Tiny fragments created by crop/rotation are removed before targets are rebuilt.
MIN_AUGMENTED_INSTANCE_AREA = max(
    4, int(round(0.000015 * IMG_SIZE * IMG_SIZE))
)
CENTER_TARGET_SIGMA_RANGE = (1.5, 5.0)

# Loss balance. These are starting values; inspect the separate logged terms.
SEMANTIC_LOSS_WEIGHT = 1.00
CENTER_LOSS_WEIGHT = 1.00
OFFSET_LOSS_WEIGHT = 1.50
BOUNDARY_LOSS_WEIGHT = 0.40
BOUNDARY_POSITIVE_WEIGHT = 4.0
BOUNDARY_DILATION_ITERATIONS = 1

# Instance decoder. Tune on validation only.
SEMANTIC_CONFIDENCE_THRESHOLD = 0.30
CENTER_CONFIDENCE_THRESHOLD = 0.12
CENTER_NMS_RADIUS = 4
MAX_CENTER_ASSIGNMENT_DISTANCE = int(round(0.12 * IMG_SIZE))
MIN_INSTANCE_AREA = max(12, int(round(0.00005 * IMG_SIZE * IMG_SIZE)))
MAX_CENTERS_PER_CLASS = 150

# Diagnostics.
PREVIEW_EVERY_N_EPOCHS = 10
PREVIEW_IMAGE_COUNT = 3
INSTANCE_EVALUATION_IOU = 0.50
EVALUATION_MAX_IMAGES = 0  # 0 means all images in the selected split.
SAVE_PERFORMANCE_DASHBOARD_EVERY_EPOCH = True
PERFORMANCE_SNAPSHOT_EVERY_N_EPOCHS = 10
RUN_FINAL_EVALUATION_AFTER_TRAINING = True
FINAL_EVALUATION_SPLIT = "val"
PERFORMANCE_PLOT_DPI = 150

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# =============================================================================
# 2. ARRAY LOADING, VALIDATION, AND TARGET GENERATION
# =============================================================================

def split_paths(split: str) -> dict[str, Path]:
    return {
        "images": ARRAY_DIR / f"X_{split}.npy",
        "semantic": ARRAY_DIR / f"Y_semantic_{split}.npy",
        "instance": ARRAY_DIR / f"Y_instance_{split}.npy",
        "center": ARRAY_DIR / f"Y_center_{split}.npy",
        "offset": ARRAY_DIR / f"Y_offset_{split}.npy",
    }


def read_rgb_image(path: Path) -> np.ndarray:
    """Read an image reliably, including paths containing spaces."""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def raw_preview_arrays_from_yolo() -> dict[str, np.ndarray]:
    """Build one preview sample directly from images/train and labels/train.

    This fallback is intentionally preview-only. It lets augmentation be
    inspected before the complete instance_npy_512 dataset has been generated.
    """
    images_root = DATASET_ROOT / "images" / "train"
    labels_root = DATASET_ROOT / "labels" / "train"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            "Neither the instance arrays nor the raw training folders exist.\n"
            f"Expected images: {images_root}\n"
            f"Expected labels: {labels_root}\n"
            "DATASET_ROOT must point to the original Split_Data folder."
        )

    image_paths = sorted(
        (
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )
    if not image_paths:
        raise FileNotFoundError(f"No training images found inside: {images_root}")

    selected_image_path: Path | None = None
    selected_label_path: Path | None = None
    for image_path in image_paths:
        relative_path = image_path.relative_to(images_root)
        label_path = labels_root / relative_path.with_suffix(".txt")
        if label_path.is_file() and label_path.stat().st_size > 0:
            selected_image_path = image_path
            selected_label_path = label_path
            break

    if selected_image_path is None or selected_label_path is None:
        raise FileNotFoundError(
            "No non-empty YOLO label file matching a training image was found in:\n"
            f"{labels_root}"
        )

    image = read_rgb_image(selected_image_path)
    image = cv2.resize(
        image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    semantic = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    instance = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)
    next_instance_id = 1

    for line_number, raw_line in enumerate(
        selected_label_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        tokens = raw_line.strip().split()
        if not tokens:
            continue
        if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
            raise ValueError(
                f"Invalid YOLO polygon at {selected_label_path}, line {line_number}."
            )
        try:
            raw_class_value = float(tokens[0])
            coordinates = np.asarray(tokens[1:], dtype=np.float64).reshape(-1, 2)
        except ValueError as error:
            raise ValueError(
                f"Non-numeric YOLO value at {selected_label_path}, "
                f"line {line_number}."
            ) from error
        if not raw_class_value.is_integer():
            raise ValueError(
                f"Non-integer class ID at {selected_label_path}, line {line_number}."
            )
        raw_class_id = int(raw_class_value)
        class_id = (
            raw_class_id
            if RAW_YOLO_CLASS_IDS_START_AT_ONE
            else raw_class_id + 1
        )
        if class_id <= 0 or class_id >= NUM_CLASSES:
            expected = "1..4" if RAW_YOLO_CLASS_IDS_START_AT_ONE else "0..3"
            raise ValueError(
                f"Class ID {raw_class_id} at {selected_label_path}, "
                f"line {line_number}; expected raw IDs {expected}."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(
                f"NaN or infinity at {selected_label_path}, line {line_number}."
            )
        coordinates = np.clip(coordinates, 0.0, 1.0)
        polygon = np.rint(
            coordinates * np.asarray([IMG_SIZE - 1, IMG_SIZE - 1])
        ).astype(np.int32)
        if len(polygon) < 3:
            continue
        polygon_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], color=1)
        pixels = polygon_mask.astype(bool)
        semantic[pixels] = class_id
        instance[pixels] = next_instance_id
        next_instance_id += 1

    if next_instance_id == 1:
        raise ValueError(f"No valid polygons found in: {selected_label_path}")

    semantic, instance, center, offset = rebuild_instance_targets(
        semantic, instance
    )
    print("Raw preview image:", selected_image_path)
    print("Raw preview label:", selected_label_path)
    return {
        "images": image[np.newaxis, ...],
        "semantic": semantic[np.newaxis, ...],
        "instance": instance[np.newaxis, ...],
        "center": center[np.newaxis, ...],
        "offset": offset[np.newaxis, ...],
    }


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to finite float32 values in [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains NaN or infinity.")
    if float(image.min()) < 0.0:
        raise ValueError("Expected non-negative RGB values.")
    if float(image.max()) > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def validate_split_arrays(split: str, arrays: dict[str, np.ndarray]) -> None:
    image_shape = arrays["images"].shape
    expected_mask_shape = image_shape[:3]

    if len(image_shape) != 4 or image_shape[1:] != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(
            f"Unexpected {split} image shape {image_shape}. Expected "
            f"(N, {IMG_SIZE}, {IMG_SIZE}, 3). If only 256 arrays exist, "
            "regenerate the instance arrays at 512 or set IMG_SIZE=256."
        )

    expected_shapes = {
        "semantic": expected_mask_shape,
        "instance": expected_mask_shape,
        "center": (image_shape[0], IMG_SIZE, IMG_SIZE, NUM_CLASSES - 1),
        "offset": (image_shape[0], IMG_SIZE, IMG_SIZE, 3),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(
                f"Unexpected {split} {key} shape {arrays[key].shape}; "
                f"expected {expected}."
            )

    sample_count = min(16, image_shape[0])
    images = np.asarray(arrays["images"][:sample_count])
    semantic = np.asarray(arrays["semantic"][:sample_count])
    instance = np.asarray(arrays["instance"][:sample_count])
    center = np.asarray(arrays["center"][:sample_count])
    offset = np.asarray(arrays["offset"][:sample_count])

    for name, values in {
        "images": images,
        "semantic": semantic,
        "instance": instance,
        "center": center,
        "offset": offset,
    }.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"NaN or infinity found in {split} {name}.")

    if semantic.min() < 0 or semantic.max() >= NUM_CLASSES:
        raise ValueError(f"Invalid semantic IDs in {split}: {np.unique(semantic)}")
    if instance.min() < 0:
        raise ValueError(f"Negative instance ID found in {split}.")
    if center.min() < 0.0 or center.max() > 1.0:
        raise ValueError(f"Center heatmaps for {split} must be in [0, 1].")
    if offset[..., 2].min() < 0.0 or offset[..., 2].max() > 1.0:
        raise ValueError(f"Offset valid-mask for {split} must be in [0, 1].")

    # A foreground semantic pixel should normally belong to an instance.
    foreground = semantic > 0
    if np.any(foreground) and np.mean(instance[foreground] > 0) < 0.99:
        raise ValueError(
            f"More than 1% of sampled foreground pixels in {split} have no "
            "instance ID. Recheck create_instance_datasets.py."
        )

    print(
        f"{split}: images={image_shape}, dtype={arrays['images'].dtype}, "
        f"sample_range=({float(images.min()):.4f}, {float(images.max()):.4f})"
    )


def load_split_arrays(split: str) -> dict[str, np.ndarray]:
    """Memory-map one split and validate the complete tensor contract."""
    paths = split_paths(split)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required instance arrays are missing. Generate them at the same "
            f"IMG_SIZE={IMG_SIZE} first:\n{missing_text}"
        )

    arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    validate_split_arrays(split, arrays)
    return arrays


def make_instance_boundary(instance_map: np.ndarray) -> np.ndarray:
    """Create a boundary target from changes in true instance IDs.

    This captures object/background boundaries and touching objects that share
    the same semantic class, which a semantic-only edge target would miss.
    """
    instance_map = np.asarray(instance_map)
    boundary = np.zeros(instance_map.shape, dtype=np.uint8)

    horizontal_change = instance_map[:, 1:] != instance_map[:, :-1]
    horizontal_valid = (instance_map[:, 1:] > 0) | (instance_map[:, :-1] > 0)
    horizontal = horizontal_change & horizontal_valid
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    vertical_change = instance_map[1:, :] != instance_map[:-1, :]
    vertical_valid = (instance_map[1:, :] > 0) | (instance_map[:-1, :] > 0)
    vertical = vertical_change & vertical_valid
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical

    if BOUNDARY_DILATION_ITERATIONS > 0:
        boundary = cv2.dilate(
            boundary,
            np.ones((3, 3), dtype=np.uint8),
            iterations=BOUNDARY_DILATION_ITERATIONS,
        )

    return boundary[..., np.newaxis].astype(np.float32)


# =============================================================================
# 3. PAIRED ONLINE AUGMENTATION
# =============================================================================

def validate_augmentation_configuration() -> None:
    expected_cycle_length = (
        1 + EXACT_ROTATION_MODE_COUNT + REALISTIC_VARIANT_MODE_COUNT
    )
    if AUGMENTATION_CYCLE_LENGTH != expected_cycle_length:
        raise ValueError(
            "AUGMENTATION_CYCLE_LENGTH must equal one original mode + all "
            "exact rotation modes + all realistic variant modes: "
            f"expected {expected_cycle_length}."
        )
    if MIN_ROTATION_DEGREES <= 0:
        raise ValueError("MIN_ROTATION_DEGREES must be positive.")
    if MAX_ROTATION_DEGREES < MIN_ROTATION_DEGREES:
        raise ValueError("MAX_ROTATION_DEGREES is smaller than the minimum.")
    if ROTATION_STEP_DEGREES <= 0:
        raise ValueError("ROTATION_STEP_DEGREES must be positive.")
    expected_exact_angles = tuple(
        range(
            MIN_ROTATION_DEGREES,
            MAX_ROTATION_DEGREES + 1,
            ROTATION_STEP_DEGREES,
        )
    )
    if EXACT_ROTATION_ANGLES != expected_exact_angles:
        raise ValueError(
            "EXACT_ROTATION_ANGLES must contain every configured angle exactly."
        )
    if EXACT_ROTATION_MODE_COUNT != 45:
        raise ValueError(
            "Expected exactly 45 pure rotations: 2, 4, 6, ..., 90 degrees."
        )
    if REALISTIC_VARIANT_MODE_COUNT != 49:
        raise ValueError("The existing realistic augmentation block must keep 49 modes.")
    if ZOOM_SAFETY_FACTOR < 1.0:
        raise ValueError("ZOOM_SAFETY_FACTOR must be at least 1.0.")
    if not (0.0 <= MAX_TRANSLATION_FRACTION < 0.5):
        raise ValueError("MAX_TRANSLATION_FRACTION must be in [0, 0.5).")


def realistic_variant_name(variant_mode: int) -> str:
    """Describe one of the original 49 realistic augmentation variants."""
    if 1 <= variant_mode <= 15:
        return "rotation"
    if 16 <= variant_mode <= 23:
        return "rotation + brightness/contrast"
    if 24 <= variant_mode <= 29:
        return "rotation + gamma/exposure"
    if 30 <= variant_mode <= 35:
        return "rotation + zoom/translation"
    if 36 <= variant_mode <= 40:
        return "rotation + blur/sharpen"
    if 41 <= variant_mode <= 44:
        return "rotation + camera noise"
    if 45 <= variant_mode <= 47:
        return "rotation + colour variation"
    if 48 <= variant_mode <= 49:
        return "combined realistic augmentation"
    raise ValueError(f"Invalid realistic variant mode: {variant_mode}")


def augmentation_mode_name(mode: int) -> str:
    """Describe the exact 1 + 45 + 49 online-augmentation schedule."""
    if mode == 0:
        return "original"
    if 1 <= mode <= EXACT_ROTATION_MODE_COUNT:
        angle = EXACT_ROTATION_ANGLES[mode - 1]
        return f"exact pure rotation +{angle} degrees"
    if mode < AUGMENTATION_CYCLE_LENGTH:
        variant_mode = mode - REALISTIC_VARIANT_MODE_OFFSET
        return realistic_variant_name(variant_mode)
    raise ValueError(f"Invalid augmentation mode: {mode}")


def sample_rotation_angle(rng: np.random.Generator) -> float:
    """Sample 2, 4, 6, ..., 90 degrees, optionally in either direction."""
    choices = np.arange(
        MIN_ROTATION_DEGREES,
        MAX_ROTATION_DEGREES + 1,
        ROTATION_STEP_DEGREES,
        dtype=np.int32,
    )
    angle = float(rng.choice(choices))
    if ROTATE_BOTH_DIRECTIONS and rng.random() < 0.5:
        angle = -angle
    return angle


def zoom_to_fill_affine_matrix(
    image_width: int,
    image_height: int,
    angle_degrees: float,
    extra_zoom: float,
    allow_translation: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a same-size rotation that fills the canvas without black corners.

    The mathematically required zoom crops some outer content. That trade-off is
    unavoidable when output dimensions must remain fixed and no artificial
    corner background is allowed.
    """
    radians = math.radians(angle_degrees)
    absolute_cosine = abs(math.cos(radians))
    absolute_sine = abs(math.sin(radians))
    aspect_factor = max(
        image_width / image_height,
        image_height / image_width,
    )
    fill_zoom = (
        absolute_cosine + aspect_factor * absolute_sine
    ) * ZOOM_SAFETY_FACTOR
    total_zoom = fill_zoom * float(extra_zoom)

    matrix = cv2.getRotationMatrix2D(
        (image_width / 2.0, image_height / 2.0),
        angle_degrees,
        total_zoom,
    ).astype(np.float64)

    if allow_translation:
        # Translation is limited by the additional crop margin, preventing the
        # transformed canvas from exposing empty corner pixels.
        extra_margin = max(0.0, float(extra_zoom) - 1.0)
        maximum_x = min(
            MAX_TRANSLATION_FRACTION * image_width,
            0.45 * extra_margin * image_width,
        )
        maximum_y = min(
            MAX_TRANSLATION_FRACTION * image_height,
            0.45 * extra_margin * image_height,
        )
        matrix[0, 2] += float(rng.uniform(-maximum_x, maximum_x))
        matrix[1, 2] += float(rng.uniform(-maximum_y, maximum_y))

    return matrix


def warp_image_semantic_and_instances(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp image, semantic IDs, and instance IDs with compatible interpolation."""
    height, width = semantic.shape
    output_size = (width, height)

    warped_image = cv2.warpAffine(
        image.astype(np.float32),
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_semantic = cv2.warpAffine(
        semantic.astype(np.uint8),
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int32)
    warped_instance = cv2.warpAffine(
        instance.astype(np.float32),
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_instance = np.rint(warped_instance).astype(np.int32)
    return warped_image, warped_semantic, warped_instance


def draw_gaussian_peak(
    heatmap: np.ndarray,
    center_x: int,
    center_y: int,
    sigma: float,
) -> None:
    """Draw one clipped Gaussian using maximum composition."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x0 = max(0, center_x - radius)
    x1 = min(heatmap.shape[1], center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(heatmap.shape[0], center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    x_grid = np.arange(x0, x1, dtype=np.float32) - float(center_x)
    y_grid = np.arange(y0, y1, dtype=np.float32) - float(center_y)
    gaussian = np.exp(
        -(y_grid[:, np.newaxis] ** 2 + x_grid[np.newaxis, :] ** 2)
        / (2.0 * sigma * sigma)
    ).astype(np.float32)
    heatmap[y0:y1, x0:x1] = np.maximum(
        heatmap[y0:y1, x0:x1], gaussian
    )


def rebuild_instance_targets(
    semantic: np.ndarray,
    instance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate consistent semantic, center, and offset targets.

    Rebuilding is safer than interpolating center/offset tensors. In particular,
    the new offset vectors point to a center that remains inside a clipped or
    rotated object.
    """
    semantic = np.asarray(semantic, dtype=np.int32).copy()
    instance = np.asarray(instance, dtype=np.int32).copy()
    height, width = semantic.shape

    semantic[instance <= 0] = 0
    center = np.zeros(
        (height, width, NUM_CLASSES - 1), dtype=np.float32
    )
    offset = np.zeros((height, width, 3), dtype=np.float32)

    flat_instance = instance.reshape(-1)
    flat_semantic = semantic.reshape(-1)
    foreground_positions = np.flatnonzero(flat_instance > 0)
    if len(foreground_positions) == 0:
        return semantic, instance, center, offset

    foreground_ids = flat_instance[foreground_positions]
    order = np.argsort(foreground_ids, kind="stable")
    sorted_positions = foreground_positions[order]
    sorted_ids = foreground_ids[order]
    unique_ids, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )

    for _, start, count in zip(unique_ids, starts, counts):
        positions = sorted_positions[start : start + count]
        if int(count) < MIN_AUGMENTED_INSTANCE_AREA:
            flat_instance[positions] = 0
            flat_semantic[positions] = 0
            continue

        class_values = flat_semantic[positions]
        class_counts = np.bincount(class_values, minlength=NUM_CLASSES)
        class_counts[0] = 0
        class_id = int(np.argmax(class_counts))
        if class_id <= 0 or class_id >= NUM_CLASSES or class_counts[class_id] == 0:
            flat_instance[positions] = 0
            flat_semantic[positions] = 0
            continue

        flat_semantic[positions] = class_id
        ys = positions // width
        xs = positions % width

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        closest = int(
            np.argmin((xs - mean_x) ** 2 + (ys - mean_y) ** 2)
        )
        center_x = int(xs[closest])
        center_y = int(ys[closest])

        equivalent_radius = math.sqrt(float(count) / math.pi)
        sigma = float(
            np.clip(
                0.25 * equivalent_radius,
                CENTER_TARGET_SIGMA_RANGE[0],
                CENTER_TARGET_SIGMA_RANGE[1],
            )
        )
        draw_gaussian_peak(
            center[..., class_id - 1], center_x, center_y, sigma
        )

        offset[ys, xs, 0] = (float(center_x) - xs.astype(np.float32)) / width
        offset[ys, xs, 1] = (float(center_y) - ys.astype(np.float32)) / height
        offset[ys, xs, 2] = 1.0

    return semantic, instance, center, offset


def apply_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    brightness = float(
        rng.uniform(-BRIGHTNESS_LIMIT, BRIGHTNESS_LIMIT) * strength
    )
    contrast = float(
        1.0 + rng.uniform(-CONTRAST_LIMIT, CONTRAST_LIMIT) * strength
    )
    return np.clip(
        (image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0
    )


def apply_gamma_exposure(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    gamma_low = 1.0 + (GAMMA_RANGE[0] - 1.0) * strength
    gamma_high = 1.0 + (GAMMA_RANGE[1] - 1.0) * strength
    gain_low = 1.0 + (EXPOSURE_GAIN_RANGE[0] - 1.0) * strength
    gain_high = 1.0 + (EXPOSURE_GAIN_RANGE[1] - 1.0) * strength
    gamma = float(rng.uniform(gamma_low, gamma_high))
    gain = float(rng.uniform(gain_low, gain_high))
    return np.clip(np.power(np.clip(image, 0.0, 1.0), gamma) * gain, 0.0, 1.0)


def apply_blur_or_sharpen(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    sigma = float(rng.uniform(0.30, 0.85) * max(strength, 0.25))
    blurred = cv2.GaussianBlur(
        image, (3, 3), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
    )
    if rng.random() < 0.5:
        return np.clip(blurred, 0.0, 1.0)
    amount = float(rng.uniform(0.15, 0.35) * strength)
    return np.clip(image + amount * (image - blurred), 0.0, 1.0)


def apply_camera_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    standard_deviation = float(
        rng.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1]) * strength
    )
    noise = rng.normal(0.0, standard_deviation, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def apply_colour_variation(
    image: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
    hue_shift = float(rng.uniform(-HUE_SHIFT_DEGREES, HUE_SHIFT_DEGREES) * strength)
    saturation_low = 1.0 + (SATURATION_GAIN_RANGE[0] - 1.0) * strength
    saturation_high = 1.0 + (SATURATION_GAIN_RANGE[1] - 1.0) * strength
    saturation_gain = float(rng.uniform(saturation_low, saturation_high))
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 360.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_gain, 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def augment_training_sample(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    mode: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one exact-rotation or realistic paired online augmentation mode."""
    if mode <= 0 or mode >= AUGMENTATION_CYCLE_LENGTH:
        raise ValueError(
            f"Expected online augmentation mode 1..{AUGMENTATION_CYCLE_LENGTH - 1}, "
            f"got {mode}."
        )

    image = normalize_image(image)
    if mode <= EXACT_ROTATION_MODE_COUNT:
        # Guaranteed pure rotations: mode 1 -> 2 degrees, ..., mode 45 -> 90.
        angle = float(EXACT_ROTATION_ANGLES[mode - 1])
        variant_mode = 0
        use_translation = False
        extra_zoom = 1.0
    else:
        # Preserve the original 49 realistic variants after the exact block.
        variant_mode = mode - REALISTIC_VARIANT_MODE_OFFSET
        angle = sample_rotation_angle(rng)
        use_translation = (
            30 <= variant_mode <= 35 or 48 <= variant_mode <= 49
        )
        if 30 <= variant_mode <= 35:
            extra_zoom = float(rng.uniform(*EXTRA_ZOOM_RANGE))
        elif 48 <= variant_mode <= 49:
            extra_zoom = float(rng.uniform(1.04, EXTRA_ZOOM_RANGE[1]))
        else:
            extra_zoom = 1.0

    matrix = zoom_to_fill_affine_matrix(
        image_width=image.shape[1],
        image_height=image.shape[0],
        angle_degrees=angle,
        extra_zoom=extra_zoom,
        allow_translation=use_translation,
        rng=rng,
    )
    image, semantic, instance = warp_image_semantic_and_instances(
        image, semantic, instance, matrix
    )
    semantic, instance, center, offset = rebuild_instance_targets(
        semantic, instance
    )

    if 16 <= variant_mode <= 23:
        image = apply_brightness_contrast(image, rng)
    elif 24 <= variant_mode <= 29:
        image = apply_gamma_exposure(image, rng)
    elif 36 <= variant_mode <= 40:
        image = apply_blur_or_sharpen(image, rng)
    elif 41 <= variant_mode <= 44:
        image = apply_camera_noise(image, rng)
    elif 45 <= variant_mode <= 47:
        image = apply_colour_variation(image, rng)
    elif 48 <= variant_mode <= 49:
        image = apply_brightness_contrast(image, rng, strength=0.60)
        image = apply_gamma_exposure(image, rng, strength=0.50)
        image = apply_colour_variation(image, rng, strength=0.50)
        if rng.random() < 0.5:
            image = apply_camera_noise(image, rng, strength=0.50)
        else:
            image = apply_blur_or_sharpen(image, rng, strength=0.50)

    image = np.asarray(np.clip(image, 0.0, 1.0), dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("Online augmentation produced NaN or infinity.")
    return image, semantic, instance, center, offset


# =============================================================================
# 4. BATCH DATA LOADER
# =============================================================================


class InstanceArraySequence(tf.keras.utils.Sequence):
    """Batch loader for images and all four Model_v5 training targets."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        batch_size: int,
        training: bool,
        seed_offset: int = 0,
        shuffle: bool = True,
    ):
        super().__init__()
        self.images = arrays["images"]
        self.semantic = arrays["semantic"]
        self.instance = arrays["instance"]
        self.center = arrays["center"]
        self.offset = arrays["offset"]
        self.batch_size = int(batch_size)
        self.training = bool(training)
        self.shuffle = bool(shuffle)
        self.seed_offset = int(seed_offset)
        self.epoch_index = 0
        self.indices = np.arange(len(self.images))
        self.order_rng = np.random.default_rng(
            SEED + self.seed_offset + (1 if training else 2)
        )
        if self.training and self.shuffle:
            self.order_rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.training:
            self.epoch_index += 1
            if self.shuffle:
                self.order_rng.shuffle(self.indices)

    def augmentation_mode_for_sample(self, sample_index: int) -> int:
        """Give every image all 95 modes exactly once per 95-epoch cycle."""
        return (self.epoch_index + int(sample_index)) % AUGMENTATION_CYCLE_LENGTH

    def augmentation_rng_for_sample(
        self, sample_index: int
    ) -> np.random.Generator:
        """Make augmentation independent of batch order and worker scheduling."""
        seed_sequence = np.random.SeedSequence(
            [SEED, self.seed_offset, self.epoch_index, int(sample_index)]
        )
        return np.random.default_rng(seed_sequence)

    def __getitem__(self, batch_index: int):
        start = batch_index * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        batch_length = len(batch_indices)

        image_batch = np.empty(
            (batch_length, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32
        )
        semantic_batch = np.empty(
            (batch_length, IMG_SIZE, IMG_SIZE), dtype=np.int32
        )
        center_batch = np.empty(
            (batch_length, IMG_SIZE, IMG_SIZE, NUM_CLASSES - 1), dtype=np.float32
        )
        offset_batch = np.empty(
            (batch_length, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32
        )
        boundary_batch = np.empty(
            (batch_length, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32
        )

        for batch_position, sample_index in enumerate(batch_indices):
            image = normalize_image(self.images[sample_index])
            semantic = np.asarray(self.semantic[sample_index], dtype=np.int32).copy()
            instance = np.asarray(self.instance[sample_index], dtype=np.int32).copy()

            if self.training and USE_ONLINE_AUGMENTATION:
                mode = self.augmentation_mode_for_sample(int(sample_index))
            else:
                mode = 0

            if mode == 0:
                center = np.asarray(
                    self.center[sample_index], dtype=np.float32
                ).copy()
                offset = np.asarray(
                    self.offset[sample_index], dtype=np.float32
                ).copy()
            else:
                image, semantic, instance, center, offset = augment_training_sample(
                    image=image,
                    semantic=semantic,
                    instance=instance,
                    mode=mode,
                    rng=self.augmentation_rng_for_sample(int(sample_index)),
                )

            boundary = make_instance_boundary(instance)

            image_batch[batch_position] = image
            semantic_batch[batch_position] = semantic
            center_batch[batch_position] = center
            offset_batch[batch_position] = offset
            boundary_batch[batch_position] = boundary

        return image_batch, {
            "semantic": semantic_batch,
            "center": center_batch,
            "offset": offset_batch,
            "boundary": boundary_batch,
        }


class AugmentationEpochSyncCallback(tf.keras.callbacks.Callback):
    """Keep the deterministic augmentation cycle correct after run recovery."""

    def __init__(self, training_sequence: InstanceArraySequence):
        super().__init__()
        self.training_sequence = training_sequence

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self.training_sequence.epoch_index = int(epoch)


class PerformanceDashboardCallback(tf.keras.callbacks.Callback):
    """Save live training history and a readable performance dashboard."""

    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.history: dict[str, list[float | int]] = {"epoch": []}
        self.live_path = self.output_dir / "training_performance_live.png"
        self.final_path = self.output_dir / "training_performance_final.png"
        self.history_path = self.output_dir / "training_history_live.json"
        self.summary_path = self.output_dir / "training_performance_summary.json"

    @staticmethod
    def scalar(value) -> float | None:
        try:
            converted = float(np.asarray(value))
        except (TypeError, ValueError):
            return None
        return converted if np.isfinite(converted) else None

    def on_train_begin(self, logs=None) -> None:
        del logs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        guide = """MODEL_V5 PERFORMANCE GUIDE

Primary checkpoint metric:
  val_semantic_foreground_miou
  Higher is better. It ignores background and averages IoU over classes 1..4.

Important supporting metrics:
  val_semantic_iou_*       Per-class semantic IoU; higher is better.
  val_boundary_boundary_f1 Boundary separation F1; higher is better.
  val_loss                 Total validation loss; lower is better.

Interpret train versus validation:
  Both improve              Healthy learning.
  Train improves, val falls Overfitting.
  Neither improves          Underfitting, data, label, or optimization issue.
  Pixel accuracy high but foreground IoU low
                             Background dominance is hiding poor detection.
"""
        (self.output_dir / "metrics_guide.txt").write_text(
            guide, encoding="utf-8"
        )
        if plt is None:
            print(
                "WARNING: matplotlib is not installed. JSON, CSV and "
                "TensorBoard metrics will be saved, but PNG graphs are disabled."
            )

    def current_learning_rate(self) -> float | None:
        try:
            value = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            return self.scalar(value)
        except (AttributeError, TypeError, ValueError):
            return None

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs if logs is not None else {}
        learning_rate = self.current_learning_rate()
        if learning_rate is not None:
            logs["learning_rate"] = learning_rate

        self.history["epoch"].append(int(epoch) + 1)
        for key, value in sorted(logs.items()):
            converted = self.scalar(value)
            if converted is not None:
                self.history.setdefault(key, []).append(converted)

        self.history_path.write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        if SAVE_PERFORMANCE_DASHBOARD_EVERY_EPOCH:
            self.render_dashboard(self.live_path)
        if (
            PERFORMANCE_SNAPSHOT_EVERY_N_EPOCHS > 0
            and (int(epoch) + 1) % PERFORMANCE_SNAPSHOT_EVERY_N_EPOCHS == 0
        ):
            snapshot_dir = self.output_dir / "epoch_snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.render_dashboard(
                snapshot_dir / f"performance_epoch_{int(epoch) + 1:04d}.png"
            )
        print("Performance dashboard updated:", self.live_path)

    def history_key(self, *candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in self.history and self.history[candidate]:
                return candidate
        return None

    def plot_pair(
        self,
        axis,
        train_candidates: tuple[str, ...],
        validation_candidates: tuple[str, ...],
        title: str,
        y_limits: tuple[float, float] | None = None,
    ) -> None:
        epoch_values = self.history["epoch"]
        train_key = self.history_key(*train_candidates)
        validation_key = self.history_key(*validation_candidates)
        if train_key is not None:
            values = self.history[train_key]
            axis.plot(epoch_values[: len(values)], values, label="train", linewidth=2)
        if validation_key is not None:
            values = self.history[validation_key]
            axis.plot(epoch_values[: len(values)], values, label="validation", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        if train_key is not None or validation_key is not None:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center")

    def render_dashboard(self, output_path: Path) -> None:
        if plt is None or not self.history["epoch"]:
            return

        figure, axes = plt.subplots(4, 3, figsize=(18, 18))
        self.plot_pair(axes[0, 0], ("loss",), ("val_loss",), "Total loss")
        self.plot_pair(
            axes[0, 1], ("semantic_loss",), ("val_semantic_loss",),
            "Semantic loss",
        )
        self.plot_pair(
            axes[0, 2], ("center_loss",), ("val_center_loss",), "Center loss"
        )
        self.plot_pair(
            axes[1, 0], ("offset_loss",), ("val_offset_loss",), "Offset loss"
        )
        self.plot_pair(
            axes[1, 1], ("boundary_loss",), ("val_boundary_loss",),
            "Boundary loss",
        )
        self.plot_pair(
            axes[1, 2],
            ("semantic_foreground_miou", "foreground_miou"),
            ("val_semantic_foreground_miou", "val_foreground_miou"),
            "Foreground mean IoU (primary)",
            (0.0, 1.0),
        )
        self.plot_pair(
            axes[2, 0],
            ("semantic_mean_iou", "mean_iou"),
            ("val_semantic_mean_iou", "val_mean_iou"),
            "Mean IoU including background",
            (0.0, 1.0),
        )

        class_axis = axes[2, 1]
        class_metrics = [
            ("val_semantic_iou_rectangle", "Rectangle"),
            ("val_semantic_iou_rectangle_concave", "Rectangle concave"),
            ("val_semantic_iou_circle", "Circle"),
            ("val_semantic_iou_circle_full", "Circle full"),
        ]
        plotted_class = False
        for key, label in class_metrics:
            if key in self.history and self.history[key]:
                values = self.history[key]
                class_axis.plot(
                    self.history["epoch"][: len(values)],
                    values,
                    label=label,
                    linewidth=2,
                )
                plotted_class = True
        class_axis.set_title("Validation IoU by class")
        class_axis.set_xlabel("Epoch")
        class_axis.set_ylim(0.0, 1.0)
        class_axis.grid(alpha=0.25)
        if plotted_class:
            class_axis.legend(fontsize=7)
        else:
            class_axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center")

        last_axis = axes[2, 2]
        accuracy_key = self.history_key(
            "val_semantic_pixel_accuracy", "val_pixel_accuracy"
        )
        boundary_key = self.history_key(
            "val_boundary_boundary_f1", "val_boundary_f1"
        )
        if accuracy_key is not None:
            values = self.history[accuracy_key]
            last_axis.plot(
                self.history["epoch"][: len(values)],
                values,
                label="validation pixel accuracy",
                linewidth=2,
            )
        if boundary_key is not None:
            values = self.history[boundary_key]
            last_axis.plot(
                self.history["epoch"][: len(values)],
                values,
                label="validation boundary F1",
                linewidth=2,
            )
        last_axis.set_title("Pixel accuracy and boundary F1")
        last_axis.set_xlabel("Epoch")
        last_axis.set_ylim(0.0, 1.0)
        last_axis.grid(alpha=0.25)
        if accuracy_key is not None or boundary_key is not None:
            last_axis.legend(fontsize=8)

        learning_rate_axis = axes[3, 0]
        learning_rate_key = self.history_key("learning_rate")
        if learning_rate_key is not None:
            values = self.history[learning_rate_key]
            learning_rate_axis.plot(
                self.history["epoch"][: len(values)], values, linewidth=2
            )
            learning_rate_axis.set_yscale("log")
        else:
            learning_rate_axis.text(
                0.5, 0.5, "Learning rate not logged", ha="center", va="center"
            )
        learning_rate_axis.set_title("Learning rate")
        learning_rate_axis.set_xlabel("Epoch")
        learning_rate_axis.grid(alpha=0.25)

        train_class_axis = axes[3, 1]
        train_class_metrics = [
            ("semantic_iou_rectangle", "Rectangle"),
            ("semantic_iou_rectangle_concave", "Rectangle concave"),
            ("semantic_iou_circle", "Circle"),
            ("semantic_iou_circle_full", "Circle full"),
        ]
        plotted_train_class = False
        for key, label in train_class_metrics:
            if key in self.history and self.history[key]:
                values = self.history[key]
                train_class_axis.plot(
                    self.history["epoch"][: len(values)],
                    values,
                    label=label,
                    linewidth=2,
                )
                plotted_train_class = True
        train_class_axis.set_title("Training IoU by class")
        train_class_axis.set_xlabel("Epoch")
        train_class_axis.set_ylim(0.0, 1.0)
        train_class_axis.grid(alpha=0.25)
        if plotted_train_class:
            train_class_axis.legend(fontsize=7)

        gap_axis = axes[3, 2]
        train_foreground_key = self.history_key(
            "semantic_foreground_miou", "foreground_miou"
        )
        validation_foreground_key = self.history_key(
            "val_semantic_foreground_miou", "val_foreground_miou"
        )
        if train_foreground_key is not None and validation_foreground_key is not None:
            train_values = np.asarray(
                self.history[train_foreground_key], dtype=np.float64
            )
            validation_values = np.asarray(
                self.history[validation_foreground_key], dtype=np.float64
            )
            length = min(len(train_values), len(validation_values))
            gap = train_values[:length] - validation_values[:length]
            gap_axis.plot(
                self.history["epoch"][:length], gap, linewidth=2, color="tab:red"
            )
            gap_axis.axhline(0.0, color="black", linewidth=1)
        else:
            gap_axis.text(
                0.5, 0.5, "Foreground IoU gap unavailable", ha="center", va="center"
            )
        gap_axis.set_title("Generalization gap: train − validation FG IoU")
        gap_axis.set_xlabel("Epoch")
        gap_axis.grid(alpha=0.25)

        monitor_key = self.history_key(
            "val_semantic_foreground_miou", "val_foreground_miou"
        )
        subtitle = f"Completed epoch {self.history['epoch'][-1]}"
        if monitor_key is not None:
            values = np.asarray(self.history[monitor_key], dtype=np.float64)
            best_index = int(np.nanargmax(values))
            subtitle += (
                f" | best validation foreground mIoU={values[best_index]:.4f} "
                f"at epoch {self.history['epoch'][best_index]}"
            )
        figure.suptitle(
            "Model_v5 Training Performance\n" + subtitle,
            fontsize=16,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figure.savefig(output_path, dpi=PERFORMANCE_PLOT_DPI)
        plt.close(figure)

    def on_train_end(self, logs=None) -> None:
        del logs
        self.render_dashboard(self.final_path)
        monitor_key = self.history_key(
            "val_semantic_foreground_miou", "val_foreground_miou"
        )
        summary: dict[str, object] = {
            "epochs_completed": len(self.history["epoch"]),
            "final_epoch": self.history["epoch"][-1] if self.history["epoch"] else 0,
            "primary_metric": monitor_key,
        }
        if monitor_key is not None:
            values = np.asarray(self.history[monitor_key], dtype=np.float64)
            best_index = int(np.nanargmax(values))
            summary.update(
                {
                    "best_epoch": self.history["epoch"][best_index],
                    "best_validation_foreground_miou": float(values[best_index]),
                }
            )
        self.summary_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("Final performance dashboard:", self.final_path)
        print("Performance summary:", self.summary_path)


def semantic_class_pixel_counts(semantic_labels: np.ndarray) -> np.ndarray:
    """Count every semantic class using small memory-mapped chunks."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for start in range(0, len(semantic_labels), 32):
        chunk = np.asarray(semantic_labels[start : start + 32], dtype=np.uint8)
        counts += np.bincount(chunk.ravel(), minlength=NUM_CLASSES)
    return counts


def validate_semantic_split_for_training(
    split: str, semantic_labels: np.ndarray
) -> np.ndarray:
    """Reject empty or class-incomplete labels before expensive training."""
    counts = semantic_class_pixel_counts(semantic_labels)
    missing_classes = [
        class_id for class_id in range(1, NUM_CLASSES) if counts[class_id] == 0
    ]
    if missing_classes:
        raise ValueError(
            f"{split} arrays contain no pixels for foreground class(es) "
            f"{missing_classes}. Pixel counts: {counts.tolist()}. Do not train "
            "with background-only or mismatched labels; regenerate this split."
        )

    percentages = 100.0 * counts / max(int(counts.sum()), 1)
    print(f"{split} class validation passed:")
    for class_id in range(NUM_CLASSES):
        print(
            f"  {class_id} {CLASS_NAMES[class_id]}: {counts[class_id]:,} "
            f"pixels ({percentages[class_id]:.3f}%)"
        )
    return counts


def compute_class_weights(semantic_labels: np.ndarray) -> np.ndarray:
    """Calculate stable inverse-square-root weights over the complete split."""
    counts = semantic_class_pixel_counts(semantic_labels)

    if np.any(counts == 0):
        raise ValueError(f"At least one class has no pixels: {counts.tolist()}")

    frequency = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequency)
    weights /= weights[0]
    weights = np.minimum(weights, 8.0)

    print("Class pixel counts:", counts.tolist())
    print("Class pixel frequencies:", np.round(frequency, 6).tolist())
    print("Class weights:", np.round(weights, 3).tolist())
    return weights.astype(np.float32)


# =============================================================================
# 4. NUMERICALLY STABLE LOSSES
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class SemanticFocalDiceFromLogits(tf.keras.losses.Loss):
    """Weighted focal cross-entropy plus present-class foreground Dice."""

    def __init__(
        self,
        class_weights,
        gamma: float = 2.0,
        focal_ratio: float = 0.60,
        name: str = "semantic_focal_dice",
    ):
        super().__init__(name=name)
        self.class_weights = [float(value) for value in class_weights]
        self.gamma = float(gamma)
        self.focal_ratio = float(focal_ratio)

    def call(self, y_true, logits):
        y_true = tf.cast(y_true, tf.int32)
        if y_true.shape.rank == logits.shape.rank:
            y_true = tf.squeeze(y_true, axis=-1)

        logits = tf.cast(logits, tf.float32)
        one_hot = tf.one_hot(y_true, NUM_CLASSES, dtype=tf.float32)
        log_probabilities = tf.nn.log_softmax(logits, axis=-1)
        probabilities = tf.exp(log_probabilities)

        true_probability = tf.reduce_sum(one_hot * probabilities, axis=-1)
        true_log_probability = tf.reduce_sum(one_hot * log_probabilities, axis=-1)
        weights = tf.constant(self.class_weights, dtype=tf.float32)
        pixel_weights = tf.reduce_sum(one_hot * weights, axis=-1)

        focal_pixels = (
            -pixel_weights
            * tf.pow(1.0 - true_probability, self.gamma)
            * true_log_probability
        )
        focal_loss = tf.reduce_sum(focal_pixels) / (
            tf.reduce_sum(pixel_weights) + 1e-6
        )

        foreground_true = one_hot[..., 1:]
        foreground_pred = probabilities[..., 1:]
        reduce_axes = tuple(range(foreground_true.shape.rank - 1))
        intersection = tf.reduce_sum(
            foreground_true * foreground_pred, axis=reduce_axes
        )
        target_sum = tf.reduce_sum(foreground_true, axis=reduce_axes)
        prediction_sum = tf.reduce_sum(foreground_pred, axis=reduce_axes)
        dice = (2.0 * intersection + 1e-6) / (
            target_sum + prediction_sum + 1e-6
        )
        present = tf.cast(target_sum > 0.0, tf.float32)
        dice_loss = tf.reduce_sum((1.0 - dice) * present) / (
            tf.reduce_sum(present) + 1e-6
        )

        return self.focal_ratio * focal_loss + (1.0 - self.focal_ratio) * dice_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_weights": self.class_weights,
                "gamma": self.gamma,
                "focal_ratio": self.focal_ratio,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class CenterNetFocalFromLogits(tf.keras.losses.Loss):
    """CenterNet-style focal loss with finite empty-image normalization."""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        name: str = "centernet_focal",
    ):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def call(self, y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        probabilities = tf.clip_by_value(tf.nn.sigmoid(logits), 1e-6, 1.0 - 1e-6)

        positive_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        negative_mask = tf.cast(tf.less(y_true, 1.0), tf.float32)
        negative_weight = tf.pow(1.0 - y_true, self.beta)

        positive_loss = (
            tf.math.log(probabilities)
            * tf.pow(1.0 - probabilities, self.alpha)
            * positive_mask
        )
        negative_loss = (
            tf.math.log(1.0 - probabilities)
            * tf.pow(probabilities, self.alpha)
            * negative_weight
            * negative_mask
        )

        positive_count = tf.reduce_sum(positive_mask)
        loss_sum = -(tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss))
        normal_with_positives = loss_sum / tf.maximum(positive_count, 1.0)
        negative_only = -tf.reduce_mean(negative_loss)
        return tf.where(positive_count > 0.0, normal_with_positives, negative_only)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "beta": self.beta})
        return config


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class MaskedOffsetHuberLoss(tf.keras.losses.Loss):
    """Smooth L1/Huber loss calculated only on valid foreground pixels."""

    def __init__(self, delta: float = 0.05, name: str = "masked_offset_huber"):
        super().__init__(name=name)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        target = y_true[..., :2]
        valid = y_true[..., 2:3]
        absolute_error = tf.abs(target - y_pred)
        quadratic = tf.minimum(absolute_error, self.delta)
        linear = absolute_error - quadratic
        smooth_l1 = 0.5 * tf.square(quadratic) / self.delta + linear
        smooth_l1 = smooth_l1 * valid
        return tf.reduce_sum(smooth_l1) / (
            2.0 * tf.reduce_sum(valid) + 1e-6
        )

    def get_config(self):
        config = super().get_config()
        config.update({"delta": self.delta})
        return config


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class BoundaryBCEDiceFromLogits(tf.keras.losses.Loss):
    """Positive-weighted boundary BCE plus Dice overlap."""

    def __init__(
        self,
        positive_weight: float = BOUNDARY_POSITIVE_WEIGHT,
        name: str = "boundary_bce_dice",
    ):
        super().__init__(name=name)
        self.positive_weight = float(positive_weight)

    def call(self, y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        bce = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=self.positive_weight,
        )
        bce_loss = tf.reduce_mean(bce)

        probabilities = tf.nn.sigmoid(logits)
        intersection = tf.reduce_sum(y_true * probabilities)
        dice = (2.0 * intersection + 1e-6) / (
            tf.reduce_sum(y_true) + tf.reduce_sum(probabilities) + 1e-6
        )
        return 0.5 * bce_loss + 0.5 * (1.0 - dice)

    def get_config(self):
        config = super().get_config()
        config.update({"positive_weight": self.positive_weight})
        return config


# =============================================================================
# 5. METRICS THAT UNDERSTAND LOGITS AND FOREGROUND QUALITY
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class SparseMeanIoUFromLogits(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes: int = NUM_CLASSES, name: str = "mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, logits, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        if y_true.shape.rank == logits.shape.rank:
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class ForegroundMeanIoUFromLogits(tf.keras.metrics.Metric):
    """Mean IoU over foreground classes 1..4, excluding background."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        name: str = "foreground_miou",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.confusion = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, logits, sample_weight=None):
        del sample_weight
        y_true = tf.cast(y_true, tf.int32)
        if y_true.shape.rank == logits.shape.rank:
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        matrix = tf.math.confusion_matrix(
            tf.reshape(y_true, [-1]),
            tf.reshape(y_pred, [-1]),
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.confusion.assign_add(matrix)

    def result(self):
        true_positive = tf.linalg.diag_part(self.confusion)
        ground_truth = tf.reduce_sum(self.confusion, axis=1)
        predicted = tf.reduce_sum(self.confusion, axis=0)
        denominator = ground_truth + predicted - true_positive
        iou = tf.math.divide_no_nan(true_positive, denominator)
        foreground_iou = iou[1:]
        foreground_valid = tf.cast(denominator[1:] > 0.0, tf.float32)
        return tf.math.divide_no_nan(
            tf.reduce_sum(foreground_iou * foreground_valid),
            tf.reduce_sum(foreground_valid),
        )

    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class ClassIoUFromLogits(tf.keras.metrics.Metric):
    """Intersection-over-union for one semantic class from logits."""

    def __init__(self, class_id: int, name: str | None = None, **kwargs):
        metric_name = name or f"iou_class_{class_id}"
        super().__init__(name=metric_name, **kwargs)
        self.class_id = int(class_id)
        self.true_positive = self.add_weight(name="tp", initializer="zeros")
        self.false_positive = self.add_weight(name="fp", initializer="zeros")
        self.false_negative = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, logits, sample_weight=None):
        del sample_weight
        y_true = tf.cast(y_true, tf.int32)
        if y_true.shape.rank == logits.shape.rank:
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32)

        target = tf.equal(y_true, self.class_id)
        predicted = tf.equal(y_pred, self.class_id)
        self.true_positive.assign_add(
            tf.reduce_sum(tf.cast(target & predicted, tf.float32))
        )
        self.false_positive.assign_add(
            tf.reduce_sum(tf.cast((~target) & predicted, tf.float32))
        )
        self.false_negative.assign_add(
            tf.reduce_sum(tf.cast(target & (~predicted), tf.float32))
        )

    def result(self):
        return tf.math.divide_no_nan(
            self.true_positive,
            self.true_positive + self.false_positive + self.false_negative,
        )

    def reset_state(self):
        self.true_positive.assign(0.0)
        self.false_positive.assign(0.0)
        self.false_negative.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"class_id": self.class_id})
        return config


@tf.keras.utils.register_keras_serializable(package="pcb_instance_v5")
class BoundaryF1FromLogits(tf.keras.metrics.Metric):
    def __init__(self, name: str = "boundary_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positive = self.add_weight(name="tp", initializer="zeros")
        self.false_positive = self.add_weight(name="fp", initializer="zeros")
        self.false_negative = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, logits, sample_weight=None):
        del sample_weight
        target = tf.cast(y_true >= 0.5, tf.bool)
        predicted = tf.cast(logits >= 0.0, tf.bool)
        self.true_positive.assign_add(
            tf.reduce_sum(tf.cast(target & predicted, tf.float32))
        )
        self.false_positive.assign_add(
            tf.reduce_sum(tf.cast((~target) & predicted, tf.float32))
        )
        self.false_negative.assign_add(
            tf.reduce_sum(tf.cast(target & (~predicted), tf.float32))
        )

    def result(self):
        return tf.math.divide_no_nan(
            2.0 * self.true_positive,
            2.0 * self.true_positive + self.false_positive + self.false_negative,
        )

    def reset_state(self):
        self.true_positive.assign(0.0)
        self.false_positive.assign(0.0)
        self.false_negative.assign(0.0)


# =============================================================================
# 6. MODEL BLOCKS: GROUP NORM, C3K2, SPPF, FPN/PAN, SKIP DECODER
# =============================================================================

def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def Conv(x, filters: int, k: int = 3, s: int = 1, name: str | None = None):
    x = layers.Conv2D(
        filters,
        k,
        strides=s,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = layers.GroupNormalization(
        groups=group_count(filters),
        axis=-1,
        epsilon=1e-5,
        name=None if name is None else f"{name}_gn",
    )(x)
    return layers.Activation(
        "swish", name=None if name is None else f"{name}_silu"
    )(x)


def Bottleneck(x, filters: int, name: str | None = None):
    shortcut = x
    y = Conv(x, filters, 3, 1, name=None if name is None else f"{name}_1")
    y = Conv(y, filters, 3, 1, name=None if name is None else f"{name}_2")
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv(
            shortcut,
            filters,
            1,
            1,
            name=None if name is None else f"{name}_shortcut",
        )
    return layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])


def C3k2(x, filters: int, n: int = 2, name: str | None = None):
    branch_1 = Conv(
        x, filters // 2, 1, name=None if name is None else f"{name}_branch1"
    )
    branch_2 = Conv(
        x, filters // 2, 1, name=None if name is None else f"{name}_branch2"
    )
    for index in range(n):
        branch_2 = Bottleneck(
            branch_2,
            filters // 2,
            name=None if name is None else f"{name}_bottleneck{index + 1}",
        )
    merged = layers.Concatenate(
        name=None if name is None else f"{name}_concat"
    )([branch_1, branch_2])
    return Conv(
        merged, filters, 1, name=None if name is None else f"{name}_out"
    )


def SPPF(x, filters: int, name: str = "sppf"):
    hidden = Conv(x, filters // 2, 1, name=f"{name}_reduce")
    pooled_1 = layers.MaxPooling2D(5, 1, padding="same", name=f"{name}_pool1")(
        hidden
    )
    pooled_2 = layers.MaxPooling2D(5, 1, padding="same", name=f"{name}_pool2")(
        pooled_1
    )
    pooled_3 = layers.MaxPooling2D(5, 1, padding="same", name=f"{name}_pool3")(
        pooled_2
    )
    merged = layers.Concatenate(name=f"{name}_concat")(
        [hidden, pooled_1, pooled_2, pooled_3]
    )
    return Conv(merged, filters, 1, name=f"{name}_out")


def ChannelAttention(x, reduction: int = 8, name: str = "channel_attention"):
    channels = int(x.shape[-1])
    hidden = max(channels // reduction, 8)
    dense_1 = layers.Dense(
        hidden,
        activation="swish",
        kernel_initializer="he_normal",
        name=f"{name}_dense1",
    )
    dense_2 = layers.Dense(
        channels,
        kernel_initializer="glorot_uniform",
        name=f"{name}_dense2",
    )
    average = dense_2(dense_1(layers.GlobalAveragePooling2D()(x)))
    maximum = dense_2(dense_1(layers.GlobalMaxPooling2D()(x)))
    attention = layers.Activation("sigmoid", name=f"{name}_sigmoid")(
        layers.Add()([average, maximum])
    )
    attention = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(
        attention
    )
    return layers.Multiply(name=f"{name}_scale")([x, attention])


def upsample(x, factor: int = 2, name: str | None = None):
    return layers.UpSampling2D(
        factor,
        interpolation="bilinear",
        name=name,
    )(x)


def build_model_v5_instance(input_shape=(IMG_SIZE, IMG_SIZE, 3)) -> Model:
    inputs = layers.Input(shape=input_shape, name="image")

    # Full-resolution detail skip, then backbone P1/2 to P5/32.
    detail = Conv(inputs, 16, 3, 1, name="detail")
    p1 = Conv(detail, 32, 3, 2, name="backbone_p1")
    p2 = C3k2(Conv(p1, 64, 3, 2, name="backbone_p2_down"), 64, 2, "p2")
    p3 = C3k2(Conv(p2, 128, 3, 2, name="backbone_p3_down"), 128, 3, "p3")
    p4 = C3k2(Conv(p3, 256, 3, 2, name="backbone_p4_down"), 256, 3, "p4")
    p5 = C3k2(Conv(p4, 384, 3, 2, name="backbone_p5_down"), 384, 2, "p5")
    p5 = SPPF(p5, 384, name="sppf")
    p5 = ChannelAttention(p5, name="deep_attention")

    # FPN top-down semantic path.
    f4 = C3k2(
        layers.Concatenate(name="fpn_p4_lateral_concat")([upsample(p5), p4]),
        256,
        2,
        "fpn_p4",
    )
    f3 = C3k2(
        layers.Concatenate(name="fpn_p3_lateral_concat")([upsample(f4), p3]),
        128,
        2,
        "fpn_p3",
    )

    # PAN bottom-up localization path.
    pan4 = C3k2(
        layers.Concatenate(name="pan_p4_route_concat")(
            [Conv(f3, 128, 3, 2, name="pan_p4_down"), f4]
        ),
        256,
        2,
        "pan_p4",
    )
    pan5 = C3k2(
        layers.Concatenate(name="pan_p5_route_concat")(
            [Conv(pan4, 256, 3, 2, name="pan_p5_down"), p5]
        ),
        384,
        2,
        "pan_p5",
    )

    # Fuse P3/P4/P5 at 1/8 resolution, then recover detail with true skips.
    context_p3 = Conv(f3, 128, 1, name="context_p3")
    context_p4 = upsample(Conv(pan4, 128, 1, name="context_p4"), 2)
    context_p5 = upsample(Conv(pan5, 128, 1, name="context_p5"), 4)
    context = C3k2(
        layers.Concatenate(name="context_concat")(
            [context_p3, context_p4, context_p5]
        ),
        160,
        2,
        "context_fusion",
    )
    context = ChannelAttention(context, name="context_attention")

    decoder_p2 = C3k2(
        layers.Concatenate(name="decoder_p2_skip_concat")(
            [upsample(context, 2), p2]
        ),
        96,
        2,
        "decoder_p2",
    )
    decoder_p1 = C3k2(
        layers.Concatenate(name="decoder_p1_skip_concat")(
            [upsample(decoder_p2, 2), p1]
        ),
        64,
        2,
        "decoder_p1",
    )
    decoder_full = C3k2(
        layers.Concatenate(name="decoder_full_skip_concat")(
            [upsample(decoder_p1, 2), detail]
        ),
        40,
        2,
        "decoder_full",
    )
    shared = Conv(decoder_full, 40, 3, 1, name="shared_head_features")
    shared = layers.SpatialDropout2D(0.10, name="shared_spatial_dropout")(shared)

    semantic_features = Conv(shared, 48, 3, 1, name="semantic_features")
    semantic_logits = layers.Conv2D(
        NUM_CLASSES,
        1,
        padding="same",
        kernel_initializer="glorot_uniform",
        dtype="float32",
        name="semantic",
    )(semantic_features)

    center_features = Conv(shared, 40, 3, 1, name="center_features")
    center_logits = layers.Conv2D(
        NUM_CLASSES - 1,
        1,
        padding="same",
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.keras.initializers.Constant(-2.19),
        dtype="float32",
        name="center",
    )(center_features)

    offset_features = Conv(shared, 40, 3, 1, name="offset_features")
    offset = layers.Conv2D(
        2,
        1,
        padding="same",
        activation="tanh",
        kernel_initializer="glorot_uniform",
        dtype="float32",
        name="offset",
    )(offset_features)

    boundary_features = Conv(shared, 32, 3, 1, name="boundary_features")
    boundary_logits = layers.Conv2D(
        1,
        1,
        padding="same",
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.keras.initializers.Constant(-2.19),
        dtype="float32",
        name="boundary",
    )(boundary_features)

    return Model(
        inputs=inputs,
        outputs={
            "semantic": semantic_logits,
            "center": center_logits,
            "offset": offset,
            "boundary": boundary_logits,
        },
        name="Model_v5_PCB_Instance",
    )


# =============================================================================
# 7. TRAINING UTILITIES AND VISUAL DIAGNOSTICS
# =============================================================================

def configure_runtime() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow version:", tf.__version__)
    print("GPUs found:", gpus)
    if REQUIRE_GPU and not gpus:
        raise RuntimeError(
            "No TensorFlow GPU was found. Run inside the WSL TensorFlow "
            "environment where your NVIDIA RTX PRO 1000 is visible."
        )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if ENABLE_OP_DETERMINISM:
        try:
            tf.config.experimental.enable_op_determinism()
            print("Deterministic TensorFlow operations enabled.")
        except (AttributeError, RuntimeError) as error:
            print("Could not enable deterministic operations:", error)


def scratch_learning_rate(epoch: int, current_lr: float) -> float:
    del current_lr
    if epoch < WARMUP_EPOCHS:
        return float(LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS)

    remaining = max(1, EPOCHS - WARMUP_EPOCHS - 1)
    progress = min(1.0, (epoch - WARMUP_EPOCHS) / remaining)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine)


def build_optimizer():
    """Use newer Keras features when installed, without breaking older builds."""
    signature = inspect.signature(tf.keras.optimizers.AdamW)
    supported = signature.parameters
    kwargs = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "global_clipnorm": 5.0,
    }
    if USE_EMA and "use_ema" in supported:
        kwargs["use_ema"] = True
        kwargs["ema_momentum"] = EMA_MOMENTUM
    if GRADIENT_ACCUMULATION_STEPS > 1 and "gradient_accumulation_steps" in supported:
        kwargs["gradient_accumulation_steps"] = GRADIENT_ACCUMULATION_STEPS

    optimizer = tf.keras.optimizers.AdamW(**kwargs)
    if hasattr(optimizer, "exclude_from_weight_decay"):
        optimizer.exclude_from_weight_decay(var_names=["bias", "beta", "gamma"])

    print("AdamW options enabled:", kwargs)
    if GRADIENT_ACCUMULATION_STEPS > 1 and "gradient_accumulation_steps" not in supported:
        print(
            "Warning: this Keras version has no built-in gradient accumulation. "
            "Training will continue with the physical batch size."
        )
    if USE_EMA and "use_ema" not in supported:
        print("Warning: this Keras version has no AdamW EMA support.")
    return optimizer


def colourize_semantic(labels_map: np.ndarray) -> np.ndarray:
    labels_map = np.clip(labels_map.astype(np.int32), 0, NUM_CLASSES - 1)
    return CLASS_COLOURS_RGB[labels_map]


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))
    cv2.rectangle(image_rgb, (0, 0), (image_rgb.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        image_rgb,
        title,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image_rgb


def probability_heatmap(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_target_preview(sequence: InstanceArraySequence, output_path: Path) -> None:
    images, targets = sequence[0]
    rows = []
    for index in range(min(PREVIEW_IMAGE_COUNT, len(images))):
        image = (np.clip(images[index], 0.0, 1.0) * 255.0).astype(np.uint8)
        semantic = colourize_semantic(targets["semantic"][index])
        center = probability_heatmap(np.max(targets["center"][index], axis=-1))
        boundary = probability_heatmap(targets["boundary"][index, ..., 0])
        panels = [
            add_panel_title(image, "TRAIN IMAGE"),
            add_panel_title(semantic, "GT SEMANTIC"),
            add_panel_title(center, "GT CENTER"),
            add_panel_title(boundary, "GT INSTANCE BOUNDARY"),
        ]
        rows.append(np.concatenate(panels, axis=1))
    preview = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def save_augmentation_catalog(
    arrays: dict[str, np.ndarray], output_path: Path
) -> None:
    """Save a detailed four-panel view of representative augmentation modes."""
    sample_index = 0
    for candidate in range(min(len(arrays["images"]), 100)):
        if np.any(np.asarray(arrays["semantic"][candidate]) > 0):
            sample_index = candidate
            break

    original_image = normalize_image(arrays["images"][sample_index])
    original_semantic = np.asarray(
        arrays["semantic"][sample_index], dtype=np.int32
    )
    original_instance = np.asarray(
        arrays["instance"][sample_index], dtype=np.int32
    )
    original_center = np.asarray(
        arrays["center"][sample_index], dtype=np.float32
    )

    representative_modes = [
        0,
        1,   # exact +2 degrees
        15,  # exact +30 degrees
        30,  # exact +60 degrees
        45,  # exact +90 degrees
        REALISTIC_VARIANT_MODE_OFFSET + 1,
        REALISTIC_VARIANT_MODE_OFFSET + 16,
        REALISTIC_VARIANT_MODE_OFFSET + 24,
        REALISTIC_VARIANT_MODE_OFFSET + 30,
        REALISTIC_VARIANT_MODE_OFFSET + 36,
        REALISTIC_VARIANT_MODE_OFFSET + 41,
        REALISTIC_VARIANT_MODE_OFFSET + 45,
        REALISTIC_VARIANT_MODE_OFFSET + 48,
    ]
    rows: list[np.ndarray] = []
    for mode in representative_modes:
        if mode == 0:
            image = original_image.copy()
            semantic = original_semantic.copy()
            instance = original_instance.copy()
            center = original_center.copy()
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED, 999, sample_index, mode])
            )
            image, semantic, instance, center, _ = augment_training_sample(
                original_image,
                original_semantic,
                original_instance,
                mode,
                rng,
            )

        boundary = make_instance_boundary(instance)
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        panels = [
            add_panel_title(
                image_uint8,
                f"MODE {mode}: {augmentation_mode_name(mode).upper()}",
            ),
            add_panel_title(colourize_semantic(semantic), "TRANSFORMED SEMANTIC"),
            add_panel_title(
                probability_heatmap(np.max(center, axis=-1)),
                "REBUILT CENTER TARGET",
            ),
            add_panel_title(
                probability_heatmap(boundary[..., 0]),
                "REBUILT INSTANCE BOUNDARY",
            ),
        ]
        rows.append(np.concatenate(panels, axis=1))

    catalog = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), cv2.cvtColor(catalog, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"Could not write augmentation preview: {output_path}")
    print("Online augmentation catalog:", output_path)


def make_augmentation_overlay(
    image: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
) -> np.ndarray:
    """Overlay semantic colours and white instance boundaries on an RGB image."""
    image_uint8 = (
        np.clip(normalize_image(image), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    semantic_colour = colourize_semantic(semantic)
    overlay = image_uint8.copy()
    foreground = np.asarray(semantic) > 0
    if np.any(foreground):
        blended = (
            0.58 * image_uint8.astype(np.float32)
            + 0.42 * semantic_colour.astype(np.float32)
        ).astype(np.uint8)
        overlay[foreground] = blended[foreground]

    boundary = make_instance_boundary(instance)[..., 0] > 0.5
    overlay[boundary] = (255, 255, 255)
    return overlay


def augmentation_angle_for_preview(
    mode: int,
    sample_index: int,
) -> float | None:
    """Recover the exact angle produced by the deterministic preview seed."""
    if mode == 0:
        return None
    if mode <= EXACT_ROTATION_MODE_COUNT:
        return float(EXACT_ROTATION_ANGLES[mode - 1])
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, 999, sample_index, mode])
    )
    return sample_rotation_angle(rng)


def save_all_augmentation_modes_catalog(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    mapping_path: Path,
) -> None:
    """Save one contact sheet containing the original and all 94 augmentations."""
    sample_index = 0
    for candidate in range(min(len(arrays["images"]), 100)):
        if np.any(np.asarray(arrays["semantic"][candidate]) > 0):
            sample_index = candidate
            break

    original_image = normalize_image(arrays["images"][sample_index])
    original_semantic = np.asarray(
        arrays["semantic"][sample_index], dtype=np.int32
    )
    original_instance = np.asarray(
        arrays["instance"][sample_index], dtype=np.int32
    )

    columns = 10
    tile_image_size = 230
    tile_header_height = 54
    tile_height = tile_header_height + tile_image_size
    rows = int(math.ceil(AUGMENTATION_CYCLE_LENGTH / columns))
    contact_sheet = np.full(
        (rows * tile_height, columns * tile_image_size, 3),
        225,
        dtype=np.uint8,
    )
    mode_mapping: list[dict[str, object]] = []

    for mode in range(AUGMENTATION_CYCLE_LENGTH):
        angle = augmentation_angle_for_preview(mode, sample_index)
        if mode == 0:
            image = original_image.copy()
            semantic = original_semantic.copy()
            instance = original_instance.copy()
            mode_group = "original"
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED, 999, sample_index, mode])
            )
            image, semantic, instance, _, _ = augment_training_sample(
                original_image,
                original_semantic,
                original_instance,
                mode,
                rng,
            )
            mode_group = (
                "exact_pure_rotation"
                if mode <= EXACT_ROTATION_MODE_COUNT
                else "realistic_variant"
            )

        overlay = make_augmentation_overlay(image, semantic, instance)
        overlay = cv2.resize(
            overlay,
            (tile_image_size, tile_image_size),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.zeros((tile_height, tile_image_size, 3), dtype=np.uint8)
        tile[tile_header_height:] = overlay

        if angle is None:
            first_line = f"MODE {mode:02d}  ORIGINAL"
        else:
            first_line = f"MODE {mode:02d}  ANGLE {angle:+.0f} DEG"
        second_line = augmentation_mode_name(mode).upper()
        if len(second_line) > 34:
            second_line = second_line[:31] + "..."
        cv2.putText(
            tile,
            first_line,
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            second_line,
            (6, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 225, 255),
            1,
            cv2.LINE_AA,
        )

        row = mode // columns
        column = mode % columns
        y0 = row * tile_height
        x0 = column * tile_image_size
        contact_sheet[y0 : y0 + tile_height, x0 : x0 + tile_image_size] = tile

        mode_mapping.append(
            {
                "mode": mode,
                "group": mode_group,
                "angle_degrees": angle,
                "description": augmentation_mode_name(mode),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not success:
        raise OSError(f"Could not encode complete augmentation preview: {output_path}")
    encoded.tofile(str(output_path))
    mapping_path.write_text(
        json.dumps(mode_mapping, indent=2), encoding="utf-8"
    )
    print("Complete 95-mode augmentation preview:", output_path)
    print("Complete augmentation mode mapping:", mapping_path)


class ValidationPreviewCallback(tf.keras.callbacks.Callback):
    """Save fixed validation predictions so silent failures are visible."""

    def __init__(self, arrays: dict[str, np.ndarray], output_dir: Path):
        super().__init__()
        self.arrays = arrays
        self.output_dir = output_dir
        self.indices = np.arange(min(PREVIEW_IMAGE_COUNT, len(arrays["images"])))

    def on_epoch_end(self, epoch, logs=None):
        del logs
        epoch_number = epoch + 1
        if epoch_number != 1 and epoch_number % PREVIEW_EVERY_N_EPOCHS != 0:
            return

        batch = np.stack(
            [normalize_image(self.arrays["images"][index]) for index in self.indices]
        )
        outputs = self.model(batch, training=False)
        semantic_logits = np.asarray(outputs["semantic"])
        center_probabilities = tf.nn.sigmoid(outputs["center"]).numpy()
        boundary_probabilities = tf.nn.sigmoid(outputs["boundary"]).numpy()
        predicted_semantic = np.argmax(semantic_logits, axis=-1)

        rows = []
        for position, array_index in enumerate(self.indices):
            image = (np.clip(batch[position], 0.0, 1.0) * 255.0).astype(np.uint8)
            ground_truth = colourize_semantic(
                np.asarray(self.arrays["semantic"][array_index])
            )
            prediction = colourize_semantic(predicted_semantic[position])
            center = probability_heatmap(
                np.max(center_probabilities[position], axis=-1)
            )
            boundary = probability_heatmap(boundary_probabilities[position, ..., 0])
            panels = [
                add_panel_title(image, "VALIDATION IMAGE"),
                add_panel_title(ground_truth, "GT SEMANTIC"),
                add_panel_title(prediction, "PREDICTED SEMANTIC"),
                add_panel_title(center, "PREDICTED CENTER"),
                add_panel_title(boundary, "PREDICTED BOUNDARY"),
            ]
            rows.append(np.concatenate(panels, axis=1))

        preview = np.concatenate(rows, axis=0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"epoch_{epoch_number:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print("Validation preview:", path)


def save_model_summary(model: Model, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def train() -> None:
    validate_augmentation_configuration()
    configure_runtime()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    print("Mixed-precision policy:", tf.keras.mixed_precision.global_policy())

    train_arrays = load_split_arrays("train")
    val_arrays = load_split_arrays("val")
    validate_semantic_split_for_training("validation", val_arrays["semantic"])
    class_weights = compute_class_weights(train_arrays["semantic"])

    training_image_count = int(len(train_arrays["images"]))
    print("\nONLINE AUGMENTATION")
    print("-------------------")
    print(f"Stored training images : {training_image_count:,}")
    print(f"Presentations per epoch: {training_image_count:,}")
    print(
        f"Presentations in 100 epochs: {training_image_count * 100:,}"
    )
    print("Saved augmented files  : 0")
    print("Validation augmentation: disabled")
    print("Test augmentation      : disabled")
    print(
        "Augmentation cycle     : 1 original + 45 exact rotations + "
        "49 realistic variants per image every 95 epochs"
    )
    print(
        "Guaranteed rotations   : "
        "2, 4, 6, ..., 90 degrees exactly once per cycle"
    )

    train_data = InstanceArraySequence(
        train_arrays, BATCH_SIZE, training=True, seed_offset=100
    )
    val_data = InstanceArraySequence(
        val_arrays, BATCH_SIZE, training=False, seed_offset=200
    )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_dir = MODEL_OUTPUT_DIR / "validation_previews"
    performance_dir = MODEL_OUTPUT_DIR / "performance"
    save_target_preview(
        InstanceArraySequence(train_arrays, BATCH_SIZE, training=False),
        MODEL_OUTPUT_DIR / "training_target_preview.png",
    )
    save_augmentation_catalog(
        train_arrays,
        MODEL_OUTPUT_DIR / "online_augmentation_catalog.png",
    )

    # Fresh graph: nothing is loaded before fit().
    model = build_model_v5_instance()
    frozen_layers = [layer.name for layer in model.layers if not layer.trainable]
    if frozen_layers:
        raise RuntimeError(f"Unexpected frozen layers: {frozen_layers}")

    optimizer = build_optimizer()
    model.compile(
        optimizer=optimizer,
        loss={
            "semantic": SemanticFocalDiceFromLogits(class_weights),
            "center": CenterNetFocalFromLogits(),
            "offset": MaskedOffsetHuberLoss(),
            "boundary": BoundaryBCEDiceFromLogits(),
        },
        loss_weights={
            "semantic": SEMANTIC_LOSS_WEIGHT,
            "center": CENTER_LOSS_WEIGHT,
            "offset": OFFSET_LOSS_WEIGHT,
            "boundary": BOUNDARY_LOSS_WEIGHT,
        },
        metrics={
            "semantic": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"),
                SparseMeanIoUFromLogits(name="mean_iou"),
                ForegroundMeanIoUFromLogits(name="foreground_miou"),
                ClassIoUFromLogits(1, name="iou_rectangle"),
                ClassIoUFromLogits(2, name="iou_rectangle_concave"),
                ClassIoUFromLogits(3, name="iou_circle"),
                ClassIoUFromLogits(4, name="iou_circle_full"),
            ],
            "boundary": [BoundaryF1FromLogits()],
        },
        jit_compile=False,
    )

    print("Scratch initialization confirmed: no checkpoint was loaded.")
    print("Parameters:", f"{model.count_params():,}")
    model.summary()
    save_model_summary(model, MODEL_OUTPUT_DIR / "model_summary.txt")
    model.save_weights(MODEL_OUTPUT_DIR / "initial_random.weights.h5")

    config = {
        "model": "Model_v5_PCB_Instance",
        "training": "from_scratch",
        "dataset_root": str(DATASET_ROOT),
        "array_dir": str(ARRAY_DIR),
        "image_size": IMG_SIZE,
        "classes": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps_requested": GRADIENT_ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "early_stopping_start_epoch": EARLY_STOPPING_START_EPOCH,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "use_ema_requested": USE_EMA,
        "ema_momentum": EMA_MOMENTUM,
        "mixed_precision": USE_MIXED_PRECISION,
        "performance_reporting": {
            "live_dashboard_every_epoch": SAVE_PERFORMANCE_DASHBOARD_EVERY_EPOCH,
            "dashboard_snapshot_every_n_epochs": (
                PERFORMANCE_SNAPSHOT_EVERY_N_EPOCHS
            ),
            "tensorboard_directory": str(MODEL_OUTPUT_DIR / "tensorboard"),
            "validation_preview_every_n_epochs": PREVIEW_EVERY_N_EPOCHS,
            "per_class_iou_metrics": True,
            "final_evaluation_after_training": RUN_FINAL_EVALUATION_AFTER_TRAINING,
            "final_evaluation_split": FINAL_EVALUATION_SPLIT,
        },
        "online_augmentation": {
            "enabled": USE_ONLINE_AUGMENTATION,
            "stored_augmented_files": 0,
            "cycle_length_epochs": AUGMENTATION_CYCLE_LENGTH,
            "original_modes": 1,
            "exact_pure_rotation_modes": EXACT_ROTATION_MODE_COUNT,
            "realistic_variant_modes": REALISTIC_VARIANT_MODE_COUNT,
            "augmented_modes": (
                EXACT_ROTATION_MODE_COUNT + REALISTIC_VARIANT_MODE_COUNT
            ),
            "exact_rotation_degrees": {
                "minimum": MIN_ROTATION_DEGREES,
                "maximum": MAX_ROTATION_DEGREES,
                "step": ROTATION_STEP_DEGREES,
                "angles": list(EXACT_ROTATION_ANGLES),
                "direction": "positive angles exactly as supplied",
                "guarantee": "every image receives every angle once per cycle",
                "background": "zoom-to-fill; no black corners",
            },
            "realistic_variant_rotation_degrees": {
                "minimum": MIN_ROTATION_DEGREES,
                "maximum": MAX_ROTATION_DEGREES,
                "step": ROTATION_STEP_DEGREES,
                "both_directions": ROTATE_BOTH_DIRECTIONS,
            },
            "brightness_limit": BRIGHTNESS_LIMIT,
            "contrast_limit": CONTRAST_LIMIT,
            "gamma_range": GAMMA_RANGE,
            "exposure_gain_range": EXPOSURE_GAIN_RANGE,
            "extra_zoom_range": EXTRA_ZOOM_RANGE,
            "maximum_translation_fraction": MAX_TRANSLATION_FRACTION,
            "noise_standard_deviation_range": NOISE_STD_RANGE,
            "hue_shift_degrees": HUE_SHIFT_DEGREES,
            "saturation_gain_range": SATURATION_GAIN_RANGE,
            "validation_augmented": False,
            "test_augmented": False,
        },
        "class_weights": class_weights.tolist(),
        "loss_weights": {
            "semantic": SEMANTIC_LOSS_WEIGHT,
            "center": CENTER_LOSS_WEIGHT,
            "offset": OFFSET_LOSS_WEIGHT,
            "boundary": BOUNDARY_LOSS_WEIGHT,
        },
        "seed": SEED,
        "tensorflow_version": tf.__version__,
    }
    (MODEL_OUTPUT_DIR / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_path = MODEL_OUTPUT_DIR / "best_model_v5_instance.keras"
    final_path = MODEL_OUTPUT_DIR / "final_best_model_v5_instance.keras"
    monitor = "val_semantic_foreground_miou"

    performance_callback = PerformanceDashboardCallback(performance_dir)
    callbacks: list[tf.keras.callbacks.Callback] = [
        AugmentationEpochSyncCallback(train_data)
    ]
    optimizer_supports_ema = "use_ema" in inspect.signature(
        tf.keras.optimizers.AdamW
    ).parameters
    if USE_EMA and optimizer_supports_ema and hasattr(
        tf.keras.callbacks, "SwapEMAWeights"
    ):
        callbacks.append(tf.keras.callbacks.SwapEMAWeights(swap_on_epoch=True))

    early_stopping_kwargs = {
        "monitor": monitor,
        "mode": "max",
        "patience": EARLY_STOPPING_PATIENCE,
        "min_delta": 1e-4,
        "restore_best_weights": False,
        "verbose": 1,
    }
    if "start_from_epoch" in inspect.signature(
        tf.keras.callbacks.EarlyStopping
    ).parameters:
        early_stopping_kwargs["start_from_epoch"] = EARLY_STOPPING_START_EPOCH
    else:
        # Compatibility fallback for older TensorFlow/Keras versions.
        early_stopping_kwargs["patience"] = max(
            EARLY_STOPPING_PATIENCE,
            EARLY_STOPPING_START_EPOCH,
        )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        **early_stopping_kwargs
    )

    # These callbacks intentionally come after SwapEMAWeights so validation
    # previews and the saved best model use EMA weights when available.
    callbacks.extend(
        [
            ValidationPreviewCallback(val_arrays, preview_dir),
            tf.keras.callbacks.ModelCheckpoint(
                best_path,
                monitor=monitor,
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            early_stopping_callback,
            tf.keras.callbacks.LearningRateScheduler(
                scratch_learning_rate, verbose=1
            ),
            performance_callback,
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(
                MODEL_OUTPUT_DIR / "training_log.csv", append=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=MODEL_OUTPUT_DIR / "tensorboard",
                histogram_freq=0,
                write_steps_per_second=True,
                profile_batch=0,
            ),
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=MODEL_OUTPUT_DIR / "fit_backup",
                save_freq="epoch",
                delete_checkpoint=True,
            ),
        ]
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    if not best_path.is_file():
        raise RuntimeError(
            f"Best checkpoint was not created. Check that metric '{monitor}' "
            f"appears in history keys: {sorted(history.history)}"
        )

    # The file selected by foreground mIoU is the authoritative final model.
    best_model = tf.keras.models.load_model(best_path, compile=False)
    best_model.save(final_path)

    print("\nTraining complete.")
    print("Target preview:", MODEL_OUTPUT_DIR / "training_target_preview.png")
    print(
        "Augmentation preview:",
        MODEL_OUTPUT_DIR / "online_augmentation_catalog.png",
    )
    print("Best model:", best_path)
    print("Final copy of best model:", final_path)
    print("Training CSV:", MODEL_OUTPUT_DIR / "training_log.csv")
    print("Validation previews:", preview_dir)
    print("Live performance dashboard:", performance_callback.live_path)
    print("Final performance dashboard:", performance_callback.final_path)
    print("TensorBoard logs:", MODEL_OUTPUT_DIR / "tensorboard")
    print(
        "TensorBoard command: tensorboard --logdir "
        f"'{MODEL_OUTPUT_DIR / 'tensorboard'}' --port 6006"
    )

    if RUN_FINAL_EVALUATION_AFTER_TRAINING:
        print(
            f"\nRunning final {FINAL_EVALUATION_SPLIT} semantic and "
            "instance evaluation on the best checkpoint..."
        )
        try:
            evaluate_instances(
                FINAL_EVALUATION_SPLIT,
                evaluation_model=best_model,
            )
        except Exception as error:
            error_path = performance_dir / "final_evaluation_error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            print(
                "WARNING: training and model saving completed, but final "
                "evaluation failed:",
                error,
            )
            print("Evaluation error details:", error_path)


def preview_online_augmentation() -> None:
    """Generate representative and complete catalogs without model training."""
    validate_augmentation_configuration()
    try:
        train_arrays = load_split_arrays("train")
        print("Preview source: pre-generated instance arrays")
    except FileNotFoundError as array_error:
        print("Instance arrays are not available for preview.")
        print(array_error)
        print("Falling back to raw YOLO image and polygon label...")
        train_arrays = raw_preview_arrays_from_yolo()
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    representative_path = MODEL_OUTPUT_DIR / "online_augmentation_catalog.png"
    complete_path = MODEL_OUTPUT_DIR / "online_augmentation_all_95.png"
    mapping_path = MODEL_OUTPUT_DIR / "online_augmentation_all_95_modes.json"
    save_augmentation_catalog(train_arrays, representative_path)
    save_all_augmentation_modes_catalog(
        train_arrays,
        complete_path,
        mapping_path,
    )
    print("No training was started and no augmented dataset files were created.")


# =============================================================================
# 8. INSTANCE DECODING
# =============================================================================

def find_center_peaks(heatmap: np.ndarray) -> np.ndarray:
    """Return [x, y, score] peaks after local-maximum suppression."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    kernel_size = 2 * CENTER_NMS_RADIUS + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_maximum = heatmap >= cv2.dilate(heatmap, kernel)
    ys, xs = np.where(
        local_maximum & (heatmap >= CENTER_CONFIDENCE_THRESHOLD)
    )
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    order = np.argsort(heatmap[ys, xs])[::-1]
    selected: list[tuple[float, int, int]] = []
    minimum_distance_squared = CENTER_NMS_RADIUS**2

    for index in order:
        x = int(xs[index])
        y = int(ys[index])
        score = float(heatmap[y, x])
        if all(
            (x - old_x) ** 2 + (y - old_y) ** 2 > minimum_distance_squared
            for _, old_x, old_y in selected
        ):
            selected.append((score, x, y))
        if len(selected) >= MAX_CENTERS_PER_CLASS:
            break

    return np.asarray(
        [(x, y, score) for score, x, y in selected], dtype=np.float32
    )


def write_connected_components(
    instance_map: np.ndarray,
    class_by_instance: dict[int, int],
    candidate_mask: np.ndarray,
    class_id: int,
    next_instance_id: int,
) -> int:
    number_of_labels, components = cv2.connectedComponents(
        candidate_mask.astype(np.uint8), connectivity=8
    )
    for component_id in range(1, number_of_labels):
        component = components == component_id
        if int(component.sum()) < MIN_INSTANCE_AREA:
            continue
        instance_map[component] = next_instance_id
        class_by_instance[next_instance_id] = class_id
        next_instance_id += 1
    return next_instance_id


def decode_instances(
    semantic_probabilities: np.ndarray,
    center_probabilities: np.ndarray,
    offset_vectors: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    """Decode semantic, center, and offset outputs into separate objects."""
    semantic_labels = np.argmax(semantic_probabilities, axis=-1).astype(np.uint8)
    semantic_confidence = np.max(semantic_probabilities, axis=-1).astype(np.float32)
    foreground = (
        (semantic_labels > 0)
        & (semantic_confidence >= SEMANTIC_CONFIDENCE_THRESHOLD)
    )

    height, width = semantic_labels.shape
    instance_map = np.zeros((height, width), dtype=np.uint16)
    class_by_instance: dict[int, int] = {}
    next_instance_id = 1

    for class_id in range(1, NUM_CLASSES):
        class_pixels = foreground & (semantic_labels == class_id)
        if not np.any(class_pixels):
            continue

        # Fuse center and semantic confidence to suppress centers in impossible
        # background locations.
        fused_heatmap = (
            center_probabilities[..., class_id - 1]
            * np.sqrt(np.clip(semantic_probabilities[..., class_id], 0.0, 1.0))
        )
        centers = find_center_peaks(fused_heatmap)
        if len(centers) == 0:
            next_instance_id = write_connected_components(
                instance_map,
                class_by_instance,
                class_pixels,
                class_id,
                next_instance_id,
            )
            continue

        ys, xs = np.nonzero(class_pixels)
        projected_x = xs.astype(np.float32) + offset_vectors[ys, xs, 0] * width
        projected_y = ys.astype(np.float32) + offset_vectors[ys, xs, 1] * height
        distances_squared = (
            (projected_x[:, np.newaxis] - centers[np.newaxis, :, 0]) ** 2
            + (projected_y[:, np.newaxis] - centers[np.newaxis, :, 1]) ** 2
        )
        nearest_center = np.argmin(distances_squared, axis=1)
        nearest_distance = np.sqrt(
            distances_squared[np.arange(len(xs)), nearest_center]
        )

        assigned_any = np.zeros(len(xs), dtype=bool)
        for center_index in range(len(centers)):
            selected = (
                (nearest_center == center_index)
                & (nearest_distance <= MAX_CENTER_ASSIGNMENT_DISTANCE)
            )
            if not np.any(selected):
                continue
            assigned_any |= selected
            candidate = np.zeros((height, width), dtype=np.uint8)
            candidate[ys[selected], xs[selected]] = 1
            next_instance_id = write_connected_components(
                instance_map,
                class_by_instance,
                candidate,
                class_id,
                next_instance_id,
            )

        # Pixels with unreliable offsets fall back to connected components
        # instead of being attached to a distant false center.
        if np.any(~assigned_any):
            unassigned = np.zeros((height, width), dtype=np.uint8)
            unassigned[ys[~assigned_any], xs[~assigned_any]] = 1
            next_instance_id = write_connected_components(
                instance_map,
                class_by_instance,
                unassigned,
                class_id,
                next_instance_id,
            )

    return instance_map, class_by_instance, semantic_confidence


def summarise_instances(
    instance_map: np.ndarray,
    class_by_instance: dict[int, int],
    semantic_confidence: np.ndarray,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for instance_id in sorted(class_by_instance):
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        class_id = class_by_instance[instance_id]
        results.append(
            {
                "instance_id": int(instance_id),
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(semantic_confidence[mask].mean()),
                "area_pixels": int(mask.sum()),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
            }
        )
    return results


# =============================================================================
# 9. ASPECT-RATIO-SAFE PREDICTION AND FOLDER OUTPUT
# =============================================================================

def letterbox_rgb(
    image_rgb: np.ndarray,
    size: int = IMG_SIZE,
    fill_value: int = 114,
) -> tuple[np.ndarray, dict[str, int | float]]:
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), fill_value, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    metadata = {
        "original_width": width,
        "original_height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "left": left,
        "top": top,
        "scale": scale,
    }
    return canvas, metadata


def restore_map_to_original(
    array: np.ndarray,
    metadata: dict[str, int | float],
    interpolation: int,
) -> np.ndarray:
    left = int(metadata["left"])
    top = int(metadata["top"])
    resized_width = int(metadata["resized_width"])
    resized_height = int(metadata["resized_height"])
    cropped = array[top : top + resized_height, left : left + resized_width]
    return cv2.resize(
        cropped,
        (int(metadata["original_width"]), int(metadata["original_height"])),
        interpolation=interpolation,
    )


def make_overlay(
    image_rgb: np.ndarray,
    instance_map: np.ndarray,
    instances: list[dict[str, object]],
) -> np.ndarray:
    overlay = np.ascontiguousarray(image_rgb.astype(np.uint8).copy())
    for result in instances:
        instance_id = int(result["instance_id"])
        mask = instance_map == instance_id
        colour = np.random.default_rng(instance_id).integers(
            40, 256, size=3, dtype=np.uint8
        )
        overlay[mask] = (
            0.45 * overlay[mask].astype(np.float32)
            + 0.55 * colour.astype(np.float32)
        ).astype(np.uint8)

        x0, y0, x1, y1 = [int(value) for value in result["bbox_xyxy"]]
        colour_tuple = tuple(int(value) for value in colour)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colour_tuple, 2)
        text = f"{result['class_name']} {result['confidence']:.2f}"
        cv2.putText(
            overlay,
            text,
            (x0, max(18, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def unpack_model_outputs(outputs) -> dict[str, np.ndarray]:
    if isinstance(outputs, dict):
        return {key: np.asarray(value) for key, value in outputs.items()}
    if not isinstance(outputs, (list, tuple)):
        raise TypeError(f"Unexpected model output type: {type(outputs)}")
    raise TypeError(
        "This model should return a dictionary. If an older TensorFlow build "
        "returns a list, call model.predict() and zip model.output_names manually."
    )


def predict_one_image(model: Model, image_path: Path, output_dir: Path) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, metadata = letterbox_rgb(original_rgb)
    batch = normalize_image(letterboxed)[np.newaxis, ...]

    start_time = time.perf_counter()
    raw_outputs = model(batch, training=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    outputs = unpack_model_outputs(raw_outputs)

    semantic_probabilities = tf.nn.softmax(outputs["semantic"][0], axis=-1).numpy()
    center_probabilities = tf.nn.sigmoid(outputs["center"][0]).numpy()
    offset_vectors = outputs["offset"][0].astype(np.float32)
    boundary_probability = tf.nn.sigmoid(outputs["boundary"][0, ..., 0]).numpy()

    instance_map, class_by_instance, semantic_confidence = decode_instances(
        semantic_probabilities,
        center_probabilities,
        offset_vectors,
    )

    instance_original = restore_map_to_original(
        instance_map, metadata, cv2.INTER_NEAREST
    ).astype(np.uint16)
    confidence_original = restore_map_to_original(
        semantic_confidence, metadata, cv2.INTER_LINEAR
    ).astype(np.float32)
    boundary_original = restore_map_to_original(
        boundary_probability, metadata, cv2.INTER_LINEAR
    ).astype(np.float32)
    semantic_original = restore_map_to_original(
        np.argmax(semantic_probabilities, axis=-1).astype(np.uint8),
        metadata,
        cv2.INTER_NEAREST,
    ).astype(np.uint8)

    instances = summarise_instances(
        instance_original, class_by_instance, confidence_original
    )
    overlay = make_overlay(original_rgb, instance_original, instances)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    np.save(output_dir / f"{stem}_instance_ids.npy", instance_original)
    cv2.imwrite(
        str(output_dir / f"{stem}_semantic.png"), semantic_original
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_boundary.png"),
        np.clip(boundary_original * 255.0, 0, 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_instances.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    result_json = {
        "source": str(image_path),
        "model": str(MODEL_FOR_INFERENCE),
        "inference_ms_excluding_file_io": elapsed_ms,
        "instances": instances,
    }
    (output_dir / f"{stem}_instances.json").write_text(
        json.dumps(result_json, indent=2), encoding="utf-8"
    )
    print(
        f"{image_path.name}: {len(instances)} instances, "
        f"forward={elapsed_ms:.1f} ms"
    )


def find_prediction_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Prediction source not found: {source}")
    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in: {source}")
    return images


def predict() -> None:
    configure_runtime()
    if not MODEL_FOR_INFERENCE.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_FOR_INFERENCE}")
    model = tf.keras.models.load_model(MODEL_FOR_INFERENCE, compile=False)
    images = find_prediction_images(PREDICT_SOURCE)
    output_dir = MODEL_OUTPUT_DIR / "predictions"
    print(f"Predicting {len(images)} image(s). Results: {output_dir}")
    failures = []
    for index, image_path in enumerate(images, start=1):
        try:
            print(f"[{index}/{len(images)}]", end=" ")
            predict_one_image(model, image_path, output_dir)
        except Exception as error:  # Keep a folder job running and report all failures.
            failures.append({"image": str(image_path), "error": str(error)})
            print("FAILED:", error)
    if failures:
        (output_dir / "prediction_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        print(f"{len(failures)} image(s) failed; see prediction_failures.json")


# =============================================================================
# 10. INSTANCE-LEVEL VALIDATION AT IoU 0.50
# =============================================================================

def ground_truth_instance_classes(
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for instance_id in np.unique(instance_map):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        pixels = semantic_map[instance_map == instance_id]
        pixels = pixels[pixels > 0]
        if len(pixels) == 0:
            continue
        result[instance_id] = int(
            np.bincount(pixels.astype(np.int32), minlength=NUM_CLASSES).argmax()
        )
    return result


def match_instances_at_iou(
    predicted_map: np.ndarray,
    predicted_classes: dict[int, int],
    target_map: np.ndarray,
    target_classes: dict[int, int],
    threshold: float,
) -> dict[int, dict[str, int]]:
    counts = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(1, NUM_CLASSES)
    }

    for class_id in range(1, NUM_CLASSES):
        predicted_ids = [
            instance_id
            for instance_id, value in predicted_classes.items()
            if value == class_id
        ]
        target_ids = [
            instance_id
            for instance_id, value in target_classes.items()
            if value == class_id
        ]

        candidates = []
        for predicted_id in predicted_ids:
            predicted_mask = predicted_map == predicted_id
            for target_id in target_ids:
                target_mask = target_map == target_id
                intersection = int(np.count_nonzero(predicted_mask & target_mask))
                if intersection == 0:
                    continue
                union = int(np.count_nonzero(predicted_mask | target_mask))
                iou = intersection / max(union, 1)
                if iou >= threshold:
                    candidates.append((iou, predicted_id, target_id))

        matched_predictions = set()
        matched_targets = set()
        for _, predicted_id, target_id in sorted(candidates, reverse=True):
            if predicted_id in matched_predictions or target_id in matched_targets:
                continue
            matched_predictions.add(predicted_id)
            matched_targets.add(target_id)

        counts[class_id]["tp"] += len(matched_predictions)
        counts[class_id]["fp"] += len(predicted_ids) - len(matched_predictions)
        counts[class_id]["fn"] += len(target_ids) - len(matched_targets)

    return counts


def metric_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def semantic_report_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    """Calculate pixel-level metrics from a semantic confusion matrix."""
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    false_positive = predicted - true_positive
    false_negative = ground_truth - true_positive
    union = true_positive + false_positive + false_negative

    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_negative) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )

    valid_all = union > 0
    valid_foreground = union[1:] > 0
    overall_accuracy = float(
        true_positive.sum() / max(float(confusion.sum()), 1.0)
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in range(NUM_CLASSES):
        per_class[CLASS_NAMES[class_id]] = {
            "class_id": class_id,
            "support_pixels": int(ground_truth[class_id]),
            "predicted_pixels": int(predicted[class_id]),
            "iou": float(iou[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1_dice": float(f1[class_id]),
        }

    return {
        "overall_pixel_accuracy": overall_accuracy,
        "mean_iou_including_background": float(
            iou[valid_all].mean() if np.any(valid_all) else 0.0
        ),
        "foreground_mean_iou": float(
            iou[1:][valid_foreground].mean()
            if np.any(valid_foreground)
            else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
    }


def save_final_evaluation_dashboard(
    split: str,
    semantic_report: dict[str, object],
    instance_report: dict[str, object],
    output_dir: Path,
) -> Path | None:
    """Save semantic and instance metrics together in one final report figure."""
    if plt is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion = np.asarray(
        semantic_report["confusion_matrix_rows_true_columns_predicted"],
        dtype=np.float64,
    )
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion),
        where=row_sums > 0,
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 13))
    confusion_axis = axes[0, 0]
    image = confusion_axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    figure.colorbar(image, ax=confusion_axis, fraction=0.046, pad=0.04)
    labels = [CLASS_NAMES[class_id] for class_id in range(NUM_CLASSES)]
    confusion_axis.set_xticks(range(NUM_CLASSES), labels, rotation=35, ha="right")
    confusion_axis.set_yticks(range(NUM_CLASSES), labels)
    confusion_axis.set_xlabel("Predicted class")
    confusion_axis.set_ylabel("True class")
    confusion_axis.set_title("Normalized semantic confusion matrix")
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = normalized[row, column]
            confusion_axis.text(
                column,
                row,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 0.50 else "black",
                fontsize=8,
            )

    foreground_names = [CLASS_NAMES[class_id] for class_id in range(1, NUM_CLASSES)]
    x_positions = np.arange(len(foreground_names), dtype=np.float64)
    bar_width = 0.20

    semantic_axis = axes[0, 1]
    semantic_per_class = semantic_report["per_class"]
    for metric_index, (metric_key, label) in enumerate(
        (("iou", "IoU"), ("precision", "Precision"),
         ("recall", "Recall"), ("f1_dice", "F1/Dice"))
    ):
        values = [
            semantic_per_class[class_name][metric_key]
            for class_name in foreground_names
        ]
        semantic_axis.bar(
            x_positions + (metric_index - 1.5) * bar_width,
            values,
            width=bar_width,
            label=label,
        )
    semantic_axis.set_xticks(x_positions, foreground_names, rotation=25, ha="right")
    semantic_axis.set_ylim(0.0, 1.0)
    semantic_axis.set_title("Semantic performance by foreground class")
    semantic_axis.grid(axis="y", alpha=0.25)
    semantic_axis.legend(fontsize=8)

    instance_axis = axes[1, 0]
    instance_per_class = instance_report["per_class"]
    for metric_index, (metric_key, label) in enumerate(
        (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
    ):
        values = [
            instance_per_class[class_name][metric_key]
            for class_name in foreground_names
        ]
        instance_axis.bar(
            x_positions + (metric_index - 1.0) * 0.25,
            values,
            width=0.25,
            label=label,
        )
    instance_axis.set_xticks(x_positions, foreground_names, rotation=25, ha="right")
    instance_axis.set_ylim(0.0, 1.0)
    instance_axis.set_title(
        f"Instance performance by class at IoU {INSTANCE_EVALUATION_IOU:.2f}"
    )
    instance_axis.grid(axis="y", alpha=0.25)
    instance_axis.legend(fontsize=8)

    summary_axis = axes[1, 1]
    summary_axis.axis("off")
    semantic_lines = [
        "SEMANTIC SUMMARY",
        f"Pixel accuracy: {semantic_report['overall_pixel_accuracy']:.4f}",
        f"Mean IoU incl. background: "
        f"{semantic_report['mean_iou_including_background']:.4f}",
        f"Foreground mean IoU: {semantic_report['foreground_mean_iou']:.4f}",
        "",
        "INSTANCE SUMMARY",
        f"Precision: {instance_report['overall']['precision']:.4f}",
        f"Recall: {instance_report['overall']['recall']:.4f}",
        f"F1: {instance_report['overall']['f1']:.4f}",
        f"TP / FP / FN: {instance_report['overall']['tp']} / "
        f"{instance_report['overall']['fp']} / {instance_report['overall']['fn']}",
        "",
        "Primary selection metric:",
        "validation foreground mean IoU",
    ]
    summary_axis.text(
        0.04,
        0.96,
        "\n".join(semantic_lines),
        va="top",
        fontsize=12,
        family="monospace",
    )

    figure.suptitle(
        f"Model_v5 Final {split.upper()} Performance",
        fontsize=17,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path = output_dir / f"{split}_final_performance_dashboard.png"
    figure.savefig(output_path, dpi=PERFORMANCE_PLOT_DPI)
    plt.close(figure)
    return output_path


def evaluate_instances(
    split: str = "val",
    evaluation_model: tf.keras.Model | None = None,
) -> None:
    if evaluation_model is None:
        configure_runtime()
        if not MODEL_FOR_INFERENCE.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_FOR_INFERENCE}")
        model = tf.keras.models.load_model(MODEL_FOR_INFERENCE, compile=False)
    else:
        model = evaluation_model
    arrays = load_split_arrays(split)

    sample_count = len(arrays["images"])
    if EVALUATION_MAX_IMAGES > 0:
        sample_count = min(sample_count, EVALUATION_MAX_IMAGES)

    aggregate = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(1, NUM_CLASSES)
    }
    semantic_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=np.int64
    )

    for index in range(sample_count):
        image = normalize_image(arrays["images"][index])[np.newaxis, ...]
        outputs = unpack_model_outputs(model(image, training=False))
        semantic_probabilities = tf.nn.softmax(
            outputs["semantic"][0], axis=-1
        ).numpy()
        center_probabilities = tf.nn.sigmoid(outputs["center"][0]).numpy()
        offset_vectors = np.asarray(outputs["offset"][0], dtype=np.float32)
        predicted_semantic = np.argmax(
            semantic_probabilities, axis=-1
        ).astype(np.int32)

        target_semantic = np.asarray(
            arrays["semantic"][index], dtype=np.int32
        )
        encoded_pairs = (
            target_semantic.ravel() * NUM_CLASSES
            + predicted_semantic.ravel()
        )
        semantic_confusion += np.bincount(
            encoded_pairs,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        predicted_map, predicted_classes, _ = decode_instances(
            semantic_probabilities,
            center_probabilities,
            offset_vectors,
        )

        target_map = np.asarray(arrays["instance"][index])
        target_classes = ground_truth_instance_classes(target_map, target_semantic)
        image_counts = match_instances_at_iou(
            predicted_map,
            predicted_classes,
            target_map,
            target_classes,
            INSTANCE_EVALUATION_IOU,
        )
        for class_id in range(1, NUM_CLASSES):
            for key in ("tp", "fp", "fn"):
                aggregate[class_id][key] += image_counts[class_id][key]

        if (index + 1) % 25 == 0 or index + 1 == sample_count:
            print(f"Evaluated {index + 1}/{sample_count} images")

    per_class = {
        CLASS_NAMES[class_id]: metric_from_counts(aggregate[class_id])
        for class_id in range(1, NUM_CLASSES)
    }
    total_counts = {
        key: sum(aggregate[class_id][key] for class_id in aggregate)
        for key in ("tp", "fp", "fn")
    }
    instance_report = {
        "model": str(MODEL_FOR_INFERENCE),
        "split": split,
        "images_evaluated": sample_count,
        "iou_threshold": INSTANCE_EVALUATION_IOU,
        "overall": metric_from_counts(total_counts),
        "per_class": per_class,
        "decoder_thresholds": {
            "semantic_confidence": SEMANTIC_CONFIDENCE_THRESHOLD,
            "center_confidence": CENTER_CONFIDENCE_THRESHOLD,
            "center_nms_radius": CENTER_NMS_RADIUS,
            "max_assignment_distance": MAX_CENTER_ASSIGNMENT_DISTANCE,
            "minimum_instance_area": MIN_INSTANCE_AREA,
        },
    }
    semantic_report = semantic_report_from_confusion(semantic_confusion)
    semantic_report.update(
        {
            "model": str(MODEL_FOR_INFERENCE),
            "split": split,
            "images_evaluated": sample_count,
        }
    )

    evaluation_dir = MODEL_OUTPUT_DIR / "performance" / f"{split}_final_evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    instance_report_path = evaluation_dir / f"{split}_instance_evaluation.json"
    semantic_report_path = evaluation_dir / f"{split}_semantic_evaluation.json"
    instance_report_path.write_text(
        json.dumps(instance_report, indent=2), encoding="utf-8"
    )
    semantic_report_path.write_text(
        json.dumps(semantic_report, indent=2), encoding="utf-8"
    )
    dashboard_path = save_final_evaluation_dashboard(
        split,
        semantic_report,
        instance_report,
        evaluation_dir,
    )

    print("\nFINAL SEMANTIC REPORT")
    print(json.dumps(semantic_report, indent=2))
    print("\nFINAL INSTANCE REPORT")
    print(json.dumps(instance_report, indent=2))
    print("Semantic evaluation:", semantic_report_path)
    print("Instance evaluation:", instance_report_path)
    if dashboard_path is not None:
        print("Final evaluation dashboard:", dashboard_path)


# =============================================================================
# 11. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if RUN_MODE == "train":
        train()
    elif RUN_MODE == "preview_augmentation":
        preview_online_augmentation()
    elif RUN_MODE == "predict":
        predict()
    elif RUN_MODE == "evaluate":
        evaluate_instances("val")
    else:
        raise ValueError(
            "RUN_MODE must be 'train', 'preview_augmentation', "
            "'predict', or 'evaluate'."
        )