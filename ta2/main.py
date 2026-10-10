
from pathlib import Path
import csv

import cv2 as cv
import numpy as np


# configuration

pattern = (7, 7)
square_mm = 20.0

input_dir = Path("images")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png"}

criteria = (
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# image normalization

def normalize_image(image, target_size):
    height, width = image.shape[:2]

    if width > height:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        height, width = image.shape[:2]

    target_width, target_height = target_size

    ratio = width / height
    target_ratio = target_width / target_height

    if abs(ratio - target_ratio) > 0.005:
        return None

    if (width, height) != target_size:
        image = cv.resize(
            image,
            target_size,
            interpolation=(
                cv.INTER_AREA
                if width > target_width
                else cv.INTER_LINEAR
            )
        )

    return image


# chessboard corner detection

def detect_corners(gray):
    found, corners = cv.findChessboardCornersSB(
        gray,
        pattern,
        flags=(
            cv.CALIB_CB_EXHAUSTIVE
            | cv.CALIB_CB_ACCURACY
            | cv.CALIB_CB_NORMALIZE_IMAGE
        )
    )

    if not found:
        found, corners = cv.findChessboardCorners(
            gray,
            pattern,
            flags=(
                cv.CALIB_CB_ADAPTIVE_THRESH
                | cv.CALIB_CB_NORMALIZE_IMAGE
            )
        )

        if found:
            corners = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

    if not found:
        return None

    return corners.reshape(-1, 1, 2).astype(np.float32)


# create 3d reference points

object_template = np.zeros(
    (pattern[0] * pattern[1], 3),
    dtype=np.float32
)

object_template[:, :2] = (
    np.mgrid[0:pattern[0], 0:pattern[1]]
    .T.reshape(-1, 2) * square_mm
)


# load images

if not input_dir.exists():
    raise SystemExit("image directory not found")

paths = sorted(
    path for path in input_dir.iterdir()
    if path.suffix.lower() in extensions
)

if not paths:
    raise SystemExit("no images found")

object_points = []
image_points = []
valid_paths = []
valid_images = []

image_size = None

for path in paths:
    image = cv.imread(str(path))

    if image is None:
        print(f"unable to read image: {path.name}")
        continue

    height, width = image.shape[:2]

    if width > height:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        height, width = image.shape[:2]

    if image_size is None:
        image_size = (width, height)

    image = normalize_image(image, image_size)

    if image is None:
        print(f"incompatible aspect ratio: {path.name}")
        continue

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    corners = detect_corners(gray)

    if corners is None:
        print(f"pattern not found: {path.name}")
        continue

    print(f"valid image: {path.name}")

    object_points.append(object_template.copy())
    image_points.append(corners)
    valid_paths.append(path)
    valid_images.append(image)

    marked = image.copy()

    cv.drawChessboardCorners(
        marked,
        pattern,
        corners,
        True
    )

    cv.imwrite(
        str(output_dir / f"{path.stem}_corners.png"),
        marked
    )


# camera calibration

if len(valid_paths) < 6:
    raise SystemExit(
        f"only {len(valid_paths)} valid images"
    )

rms, camera_matrix, distortion, rvecs, tvecs = (
    cv.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)

np.savez(
    output_dir / "calibration.npz",
    camera_matrix=camera_matrix,
    distortion=distortion,
    rms=rms,
    image_size=np.array(image_size),
    square_mm=square_mm
)

print("\ncamera calibration results")
print(f"valid images: {len(valid_paths)}")
print(f"rms error: {rms:.4f}")
print("\ncamera matrix:")
print(camera_matrix)
print("\ndistortion coefficients:")
print(distortion)


# calculate reprojection error for each image

reprojection_rows = []
all_squared_errors = []

for path, object_pts, detected, rvec, tvec in zip(
    valid_paths,
    object_points,
    image_points,
    rvecs,
    tvecs
):
    projected, _ = cv.projectPoints(
        object_pts,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    projected = projected.reshape(-1, 2)
    observed = detected.reshape(-1, 2)

    errors = np.linalg.norm(
        projected - observed,
        axis=1
    )

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    all_squared_errors.extend(
        (errors ** 2).tolist()
    )

    reprojection_rows.append([
        path.name,
        mean_error,
        max_error
    ])

    print(
        f"reprojection error: {path.name} "
        f"mean={mean_error:.4f} "
        f"max={max_error:.4f}"
    )

with (output_dir / "reprojection_errors.csv").open(
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "image",
        "mean_error_px",
        "max_error_px"
    ])

    writer.writerows(reprojection_rows)

global_rmse = float(
    np.sqrt(np.mean(all_squared_errors))
)

print(f"\nglobal reprojection rmse: {global_rmse:.4f}")


# define known 3d points

test_points = np.array([
    [0, 0, 0],
    [6 * square_mm, 0, 0],
    [0, 6 * square_mm, 0],
    [6 * square_mm, 6 * square_mm, 0],
    [3 * square_mm, 3 * square_mm, 0]
], dtype=np.float32)

corner_indices = [0, 6, 42, 48, 24]

projection_rows = []


# undistortion and 3d to 2d projection

for path, image, detected, rvec, tvec in zip(
    valid_paths,
    valid_images,
    image_points,
    rvecs,
    tvecs
):
    # remove lens distortion

    undistorted = cv.undistort(
        image,
        camera_matrix,
        distortion
    )

    cv.imwrite(
        str(output_dir / f"{path.stem}_undistorted.png"),
        undistorted
    )

    comparison = np.hstack((
        image,
        undistorted
    ))

    cv.imwrite(
        str(output_dir / f"{path.stem}_comparison.png"),
        comparison
    )

    # project 3d points into the image

    projected, _ = cv.projectPoints(
        test_points,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    projected = projected.reshape(-1, 2)
    observed_points = detected.reshape(-1, 2)

    annotated = image.copy()

    for index, corner_index in enumerate(corner_indices):
        point = projected[index]
        observed = observed_points[corner_index]

        error = float(
            np.linalg.norm(point - observed)
        )

        projection_rows.append([
            path.name,
            index + 1,
            *test_points[index].tolist(),
            *point.tolist(),
            *observed.tolist(),
            error
        ])

        projected_pixel = tuple(
            np.round(point).astype(int)
        )

        observed_pixel = tuple(
            np.round(observed).astype(int)
        )

        # projected point in red

        cv.circle(
            annotated,
            projected_pixel,
            6,
            (0, 0, 255),
            -1
        )

        # detected point in green

        cv.circle(
            annotated,
            observed_pixel,
            9,
            (0, 255, 0),
            2
        )

        cv.putText(
            annotated,
            f"p{index + 1}",
            (
                projected_pixel[0] + 12,
                projected_pixel[1] - 12
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    cv.imwrite(
        str(output_dir / f"{path.stem}_projection.png"),
        annotated
    )


# save projection results

with (output_dir / "projection_errors.csv").open(
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "image",
        "point",
        "x_mm",
        "y_mm",
        "z_mm",
        "u_projected",
        "v_projected",
        "u_detected",
        "v_detected",
        "error_px"
    ])

    writer.writerows(projection_rows)

projection_errors = [
    row[-1] for row in projection_rows
]

print("\nprojection results")
print(
    f"mean error: "
    f"{np.mean(projection_errors):.4f} pixels"
)
print(
    f"maximum error: "
    f"{np.max(projection_errors):.4f} pixels"
)
print(
    f"\nresults saved to: {output_dir.resolve()}"
)
