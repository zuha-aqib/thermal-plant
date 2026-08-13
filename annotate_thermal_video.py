import cv2
import json
import pickle
import argparse
from pathlib import Path
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Final annotations are drawn in RED.
# OpenCV uses BGR instead of RGB.
RED = (0, 0, 255)

# Current points being selected are shown in yellow so that
# they are visually different from confirmed annotations.
YELLOW = (0, 255, 255)

# Text color.
WHITE = (255, 255, 255)

# Maximum size of the annotation window.
# The original coordinates are preserved even if the frame
# needs to be resized to fit on the monitor.
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 850

# Thickness of annotation lines.
LINE_THICKNESS = 2


# ============================================================
# GLOBAL ANNOTATION STATE
# ============================================================

# These variables are changed while the user interacts with
# the annotation window.

current_points = []
regions = []

# Default drawing mode.
# Can be changed between "polygon" and "rectangle".
drawing_mode = "polygon"

# Information needed to convert displayed coordinates back
# into coordinates from the original full-resolution frame.
display_scale = 1.0

# Original frame used for annotation.
original_frame = None


# ============================================================
# HELPER: RESIZE FRAME FOR DISPLAY
# ============================================================

def calculate_display_scale(width, height):
    """
    Calculate how much the original frame needs to be scaled
    so that it fits comfortably on the user's screen.

    IMPORTANT:
    This only changes how large the frame looks in the GUI.
    Annotation coordinates are still saved relative to the
    ORIGINAL video resolution.
    """

    width_scale = MAX_DISPLAY_WIDTH / width
    height_scale = MAX_DISPLAY_HEIGHT / height

    # Never enlarge the frame beyond its original resolution.
    return min(width_scale, height_scale, 1.0)


def original_to_display(point):
    """
    Convert an original-video coordinate to the resized
    coordinate used by the annotation window.
    """

    x, y = point

    display_x = int(x * display_scale)
    display_y = int(y * display_scale)

    return display_x, display_y


def display_to_original(x, y):
    """
    Convert a mouse click from the resized display frame
    back into the ORIGINAL video coordinate system.
    """

    original_x = int(round(x / display_scale))
    original_y = int(round(y / display_scale))

    # Clamp coordinates so they always remain inside the image.
    original_x = max(0, min(original_x, original_frame.shape[1] - 1))
    original_y = max(0, min(original_y, original_frame.shape[0] - 1))

    return original_x, original_y


# ============================================================
# HELPER: RECTANGLE -> FOUR POLYGON POINTS
# ============================================================

def rectangle_points(point1, point2):
    """
    Convert two opposite rectangle corners into four polygon
    coordinates.

    Example:

        point1 -------- point2
          |                |
          |                |
          |                |
        point4 -------- point3

    Saving rectangles as four points makes the later thermal
    processing code simpler because rectangles and polygons
    can both be converted into masks using the same method.
    """

    x1, y1 = point1
    x2, y2 = point2

    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ]


# ============================================================
# DRAW ALL ANNOTATIONS
# ============================================================

def draw_annotation_preview():
    """
    Build the image shown to the user.

    Confirmed regions:
        RED

    Current unconfirmed points:
        YELLOW

    Nothing is permanently painted onto the original frame.
    """

    # Work on a COPY so our source image remains untouched.
    frame = original_frame.copy()

    # --------------------------------------------------------
    # Draw already-confirmed regions
    # --------------------------------------------------------

    for region in regions:

        points = np.array(
            region["pixel_points"],
            dtype=np.int32
        )

        # Draw red polygon.
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=RED,
            thickness=LINE_THICKNESS
        )

        # Use the upper-left-most point approximately as the
        # location for the region name.
        min_x = int(np.min(points[:, 0]))
        min_y = int(np.min(points[:, 1]))

        text_position = (
            min_x,
            max(25, min_y - 8)
        )

        cv2.putText(
            frame,
            region["name"],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Draw points currently being selected
    # --------------------------------------------------------

    if len(current_points) > 0:

        # Draw each clicked point.
        for point in current_points:

            cv2.circle(
                frame,
                tuple(point),
                radius=5,
                color=YELLOW,
                thickness=-1
            )

        # POLYGON MODE
        if drawing_mode == "polygon":

            if len(current_points) >= 2:

                points = np.array(
                    current_points,
                    dtype=np.int32
                )

                cv2.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=YELLOW,
                    thickness=LINE_THICKNESS
                )

        # RECTANGLE MODE
        elif drawing_mode == "rectangle":

            if len(current_points) == 2:

                cv2.rectangle(
                    frame,
                    tuple(current_points[0]),
                    tuple(current_points[1]),
                    YELLOW,
                    LINE_THICKNESS
                )

    # --------------------------------------------------------
    # Add instructions at the top
    # --------------------------------------------------------

    instructions = (
        f"MODE: {drawing_mode.upper()} | "
        "Left Click: point | "
        "ENTER: confirm | "
        "BACKSPACE: undo point | "
        "P: polygon | R: rectangle | "
        "U: undo region | S: save | Q: quit"
    )

    # Black background behind instructions for readability.
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 35),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        instructions,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Resize only for display
    # --------------------------------------------------------

    if display_scale != 1.0:

        frame = cv2.resize(
            frame,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )

    return frame


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    """
    Handle mouse clicks inside the OpenCV annotation window.
    """

    global current_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Convert click from displayed image coordinates back into
    # coordinates from the original video frame.
    point = display_to_original(x, y)

    # --------------------------------------------------------
    # POLYGON MODE
    # --------------------------------------------------------

    if drawing_mode == "polygon":

        # Every click adds another polygon vertex.
        current_points.append(point)

    # --------------------------------------------------------
    # RECTANGLE MODE
    # --------------------------------------------------------

    elif drawing_mode == "rectangle":

        # A rectangle only needs two points:
        # one corner and the opposite corner.

        if len(current_points) < 2:
            current_points.append(point)

        else:
            print(
                "\nRectangle already has two corners."
                "\nPress ENTER to confirm it or BACKSPACE to edit it."
            )


# ============================================================
# CONFIRM CURRENT REGION
# ============================================================

def confirm_current_region():
    """
    Turn the current temporary annotation into a permanent ROI.
    """

    global current_points

    # --------------------------------------------------------
    # Validate annotation
    # --------------------------------------------------------

    if drawing_mode == "polygon":

        if len(current_points) < 3:

            print(
                "\nA polygon needs at least 3 points."
            )

            return

        final_points = [
            [int(x), int(y)]
            for x, y in current_points
        ]

    elif drawing_mode == "rectangle":

        if len(current_points) != 2:

            print(
                "\nA rectangle needs exactly 2 opposite corners."
            )

            return

        final_points = rectangle_points(
            current_points[0],
            current_points[1]
        )

    else:
        return

    # --------------------------------------------------------
    # Ask user for ROI / equipment name
    # --------------------------------------------------------

    suggested_name = f"ROI_{len(regions) + 1:02d}"

    print()
    name = input(
        f"Enter region/equipment name "
        f"[default: {suggested_name}]: "
    ).strip()

    if not name:
        name = suggested_name

    height, width = original_frame.shape[:2]

    # --------------------------------------------------------
    # Calculate normalized points
    # --------------------------------------------------------
    #
    # Example:
    #
    # Pixel coordinate:
    #     [640, 360]
    #
    # For a 1280x720 frame:
    #
    # Normalized:
    #     [0.5, 0.5]
    #
    # This is useful if the video later gets resized.
    # --------------------------------------------------------

    normalized_points = []

    for x, y in final_points:

        normalized_points.append([
            x / width,
            y / height
        ])

    # --------------------------------------------------------
    # Save region
    # --------------------------------------------------------

    region = {
        "id": len(regions) + 1,
        "name": name,
        "shape_type": drawing_mode,
        "pixel_points": final_points,
        "normalized_points": normalized_points
    }

    regions.append(region)

    print(
        f"Confirmed region: {name}"
    )

    print(
        f"Pixel coordinates: {final_points}"
    )

    # Clear temporary points so next ROI can be created.
    current_points = []


# ============================================================
# SAVE JSON + PKL + IMAGES
# ============================================================

def save_annotations(
    output_dir,
    video_path,
    reference_frame_number,
    fps,
    total_frames
):
    """
    Save annotation information to both JSON and PKL.

    Also save:
        - clean reference frame
        - annotated reference frame
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    height, width = original_frame.shape[:2]

    # --------------------------------------------------------
    # Master annotation structure
    # --------------------------------------------------------

    annotation_data = {

        "video": {
            "filename": Path(video_path).name,

            "width": width,
            "height": height,

            "fps": float(fps),

            "total_frames": int(total_frames),

            "reference_frame_number": int(
                reference_frame_number
            ),

            "reference_time_seconds": (
                float(reference_frame_number / fps)
                if fps > 0
                else None
            )
        },

        "regions": regions
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = output_dir / "regions.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            annotation_data,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save PKL
    # --------------------------------------------------------

    pkl_path = output_dir / "regions.pkl"

    with open(
        pkl_path,
        "wb"
    ) as file:

        pickle.dump(
            annotation_data,
            file
        )

    # --------------------------------------------------------
    # Save clean reference frame
    # --------------------------------------------------------

    reference_path = (
        output_dir /
        "reference_frame.png"
    )

    cv2.imwrite(
        str(reference_path),
        original_frame
    )

    # --------------------------------------------------------
    # Save FINAL annotated reference frame
    # --------------------------------------------------------

    annotated = original_frame.copy()

    for region in regions:

        points = np.array(
            region["pixel_points"],
            dtype=np.int32
        )

        cv2.polylines(
            annotated,
            [points],
            True,
            RED,
            LINE_THICKNESS
        )

        min_x = int(np.min(points[:, 0]))
        min_y = int(np.min(points[:, 1]))

        cv2.putText(
            annotated,
            region["name"],
            (
                min_x,
                max(25, min_y - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2,
            cv2.LINE_AA
        )

    annotated_path = (
        output_dir /
        "annotated_reference_frame.png"
    )

    cv2.imwrite(
        str(annotated_path),
        annotated
    )

    print("\nAnnotations saved successfully.")
    print(f"JSON : {json_path}")
    print(f"PKL  : {pkl_path}")
    print(f"Frame: {reference_path}")
    print(f"Image: {annotated_path}")

    return annotation_data


# ============================================================
# DRAW REGIONS ON ENTIRE VIDEO
# ============================================================

def create_annotated_video(
    video_path,
    output_path,
    regions
):
    """
    Apply the SAME fixed regions to every frame of the video.

    This does NOT calculate temperatures yet.

    It simply proves that our saved annotations are correctly
    aligned throughout the entire fixed-camera video.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # MP4 output codec.
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    frame_number = 0

    print(
        "\nApplying annotations to entire video..."
    )

    while True:

        success, frame = capture.read()

        if not success:
            break

        # ----------------------------------------------------
        # Draw every saved region
        # ----------------------------------------------------

        for region in regions:

            points = np.array(
                region["pixel_points"],
                dtype=np.int32
            )

            cv2.polylines(
                frame,
                [points],
                isClosed=True,
                color=RED,
                thickness=LINE_THICKNESS
            )

            min_x = int(
                np.min(points[:, 0])
            )

            min_y = int(
                np.min(points[:, 1])
            )

            cv2.putText(
                frame,
                region["name"],
                (
                    min_x,
                    max(25, min_y - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED,
                2,
                cv2.LINE_AA
            )

        writer.write(frame)

        frame_number += 1

        # Print progress occasionally rather than printing
        # thousands of lines for a long video.
        if frame_number % 100 == 0:

            percentage = (
                100 * frame_number / total_frames
                if total_frames > 0
                else 0
            )

            print(
                f"\rProcessed "
                f"{frame_number}/{total_frames} "
                f"frames "
                f"({percentage:.1f}%)",
                end=""
            )

    capture.release()
    writer.release()

    print(
        f"\nAnnotated video saved to:\n{output_path}"
    )


# ============================================================
# MAIN ANNOTATION GUI
# ============================================================

def annotate_video(
    video_path,
    output_dir,
    frame_number=0
):
    """
    Main annotation workflow.
    """

    global original_frame
    global display_scale
    global drawing_mode
    global current_points

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    # Ensure requested frame exists.
    frame_number = max(
        0,
        min(
            frame_number,
            total_frames - 1
        )
    )

    # Jump directly to selected frame.
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = capture.read()

    capture.release()

    if not success:

        raise RuntimeError(
            f"Could not read frame {frame_number}"
        )

    original_frame = frame

    height, width = frame.shape[:2]

    display_scale = calculate_display_scale(
        width,
        height
    )

    print("\n============================================")
    print("THERMAL VIDEO ROI ANNOTATOR")
    print("============================================")

    print(
        f"Video: {video_path.name}"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.3f}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print(
        f"Reference frame: {frame_number}"
    )

    if fps > 0:

        print(
            f"Reference time: "
            f"{frame_number / fps:.2f} seconds"
        )

    print(
        "\nControls:"
    )

    print(
        "  Left Click  -> Add point"
    )

    print(
        "  ENTER       -> Confirm current ROI"
    )

    print(
        "  BACKSPACE   -> Remove previous point"
    )

    print(
        "  P           -> Polygon mode"
    )

    print(
        "  R           -> Rectangle mode"
    )

    print(
        "  U           -> Remove last confirmed ROI"
    )

    print(
        "  C           -> Clear current unfinished ROI"
    )

    print(
        "  S           -> Save and finish"
    )

    print(
        "  Q           -> Quit without saving"
    )

    # --------------------------------------------------------
    # Create annotation window
    # --------------------------------------------------------

    window_name = "Thermal ROI Annotator"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    # --------------------------------------------------------
    # Main GUI loop
    # --------------------------------------------------------

    while True:

        preview = draw_annotation_preview()

        cv2.imshow(
            window_name,
            preview
        )

        key = cv2.waitKey(20) & 0xFF

        # ----------------------------------------------------
        # ENTER
        # ----------------------------------------------------

        if key in (10, 13):

            confirm_current_region()

        # ----------------------------------------------------
        # BACKSPACE
        # ----------------------------------------------------

        elif key == 8:

            if current_points:

                removed = current_points.pop()

                print(
                    f"Removed point: {removed}"
                )

        # ----------------------------------------------------
        # POLYGON MODE
        # ----------------------------------------------------

        elif key in (
            ord("p"),
            ord("P")
        ):

            current_points = []
            drawing_mode = "polygon"

            print(
                "\nSwitched to POLYGON mode."
            )

        # ----------------------------------------------------
        # RECTANGLE MODE
        # ----------------------------------------------------

        elif key in (
            ord("r"),
            ord("R")
        ):

            current_points = []
            drawing_mode = "rectangle"

            print(
                "\nSwitched to RECTANGLE mode."
            )

        # ----------------------------------------------------
        # UNDO LAST CONFIRMED REGION
        # ----------------------------------------------------

        elif key in (
            ord("u"),
            ord("U")
        ):

            if regions:

                removed_region = regions.pop()

                print(
                    f"\nRemoved region: "
                    f"{removed_region['name']}"
                )

        # ----------------------------------------------------
        # CLEAR CURRENT UNFINISHED REGION
        # ----------------------------------------------------

        elif key in (
            ord("c"),
            ord("C")
        ):

            current_points = []

            print(
                "\nCurrent unfinished annotation cleared."
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        elif key in (
            ord("s"),
            ord("S")
        ):

            if not regions:

                print(
                    "\nNo regions have been annotated yet."
                )

                continue

            save_annotations(
                output_dir=output_dir,
                video_path=video_path,
                reference_frame_number=frame_number,
                fps=fps,
                total_frames=total_frames
            )

            break

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        elif key in (
            ord("q"),
            ord("Q")
        ):

            print(
                "\nExiting without saving."
            )

            cv2.destroyAllWindows()

            return

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Apply saved ROIs to whole video
    # --------------------------------------------------------

    annotated_video_path = (
        output_dir /
        "annotated_video.mp4"
    )

    create_annotated_video(
        video_path=video_path,
        output_path=annotated_video_path,
        regions=regions
    )


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Annotate fixed ROIs in thermal-camera videos."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the thermal video"
    )

    parser.add_argument(
        "--output",
        default="annotated-videos",
        help=(
            "Folder where JSON, PKL, images and video "
            "will be saved"
        )
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help=(
            "Video frame number to use for annotation. "
            "Default: 0"
        )
    )

    args = parser.parse_args()

    annotate_video(
        video_path=args.video,
        output_dir=args.output,
        frame_number=args.frame
    )