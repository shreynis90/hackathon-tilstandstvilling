import os
import sys

import boto3
import laspy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from botocore.exceptions import ClientError
from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------
# 1. Command-line arguments
# ---------------------------------------------------------

if len(sys.argv) < 2:
    raise SystemExit(
        "\nUsage:\n\n"
        "poetry run python get_data.py "
        '"<copc_key>" '
        '"<bridge_name>" '
        '"<bridge_id>" '
        '"<capture_date>" '
        '"<road_reference>" '
        '"<epsg_code>" '
        '"<metadata_point_count>"'
    )


copc_key = sys.argv[1]

bridge_name = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "Unknown bridge"
)

bridge_id = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "Unknown"
)

capture_date = (
    sys.argv[4]
    if len(sys.argv) > 4
    else "Unknown"
)

road_reference = (
    sys.argv[5]
    if len(sys.argv) > 5
    else "Unknown"
)

epsg_code = (
    sys.argv[6]
    if len(sys.argv) > 6
    else "Unknown"
)

metadata_point_count = (
    sys.argv[7]
    if len(sys.argv) > 7
    else "Unknown"
)


# ---------------------------------------------------------
# 2. Saga configuration
# ---------------------------------------------------------

endpoint = os.environ[
    "saga_s3_endpoint"
]

bucket = os.environ[
    "data_bucket"
]

access_key = os.environ[
    "data_access_key"
]

secret_key = os.environ[
    "data_secret_key"
]


s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)


# ---------------------------------------------------------
# 3. Check connection
# ---------------------------------------------------------

try:
    s3.list_objects_v2(
        Bucket=bucket,
        MaxKeys=1,
    )

except ClientError as exc:
    print(
        "Saga bucket access failed:"
    )
    print(exc)
    raise


# ---------------------------------------------------------
# 4. Inspect selected COPC
# ---------------------------------------------------------

response = s3.head_object(
    Bucket=bucket,
    Key=copc_key,
)

copc_size = response[
    "ContentLength"
]


# ---------------------------------------------------------
# 5. Download locally if needed
# ---------------------------------------------------------

local_data_dir = "data"

os.makedirs(
    local_data_dir,
    exist_ok=True,
)

local_copc_path = os.path.join(
    local_data_dir,
    os.path.basename(copc_key),
)


if os.path.exists(
    local_copc_path
):
    print(
        "Using local file:",
        local_copc_path,
    )

else:
    print(
        "Downloading:",
        copc_key,
    )

    s3.download_file(
        bucket,
        copc_key,
        local_copc_path,
    )

    print(
        "Downloaded:",
        local_copc_path,
    )


# ---------------------------------------------------------
# 6. Read point cloud
# ---------------------------------------------------------

print(
    "Reading point cloud..."
)

las = laspy.read(
    local_copc_path
)

header = las.header


xyz = np.column_stack(
    (
        las.x,
        las.y,
        las.z,
    )
)


# ---------------------------------------------------------
# 7. Sample points for display
# ---------------------------------------------------------

sample_size = min(
    400_000,
    len(xyz),
)

rng = np.random.default_rng(
    seed=42
)

indices = rng.choice(
    len(xyz),
    size=sample_size,
    replace=False,
)

xyz_sample = xyz[
    indices
]


# ---------------------------------------------------------
# 8. Center geometry
# ---------------------------------------------------------

center = xyz_sample.mean(
    axis=0
)

xyz_centered = (
    xyz_sample
    - center
)


# ---------------------------------------------------------
# 9. Create Open3D point cloud
# ---------------------------------------------------------

cloud = o3d.geometry.PointCloud()

cloud.points = (
    o3d.utility.Vector3dVector(
        xyz_centered
    )
)


# ---------------------------------------------------------
# 10. RGB if available
# ---------------------------------------------------------

dimension_names = set(
    las.point_format.dimension_names
)

has_rgb = {
    "red",
    "green",
    "blue",
}.issubset(
    dimension_names
)


if has_rgb:
    colors = np.column_stack(
        (
            las.red[indices],
            las.green[indices],
            las.blue[indices],
        )
    ).astype(
        np.float64
    )

    max_color = colors.max()

    if max_color > 0:
        colors /= 65535.0

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )


# ---------------------------------------------------------
# 11. Geometry extent
# ---------------------------------------------------------

extent = (
    xyz_sample.max(axis=0)
    - xyz_sample.min(axis=0)
)


# ---------------------------------------------------------
# 12. Open3D GUI app
# ---------------------------------------------------------

app = gui.Application.instance
app.initialize()


window = app.create_window(
    "TilstandsTvilling",
    1500,
    900,
)


# ---------------------------------------------------------
# 13. Main 3D scene
# ---------------------------------------------------------

scene_widget = gui.SceneWidget()

scene_widget.scene = (
    rendering.Open3DScene(
        window.renderer
    )
)


material = rendering.MaterialRecord()

material.shader = (
    "defaultUnlit"
)

material.point_size = 2.0


scene_widget.scene.add_geometry(
    "bridge_point_cloud",
    cloud,
    material,
)


bounds = cloud.get_axis_aligned_bounding_box()

scene_widget.setup_camera(
    60,
    bounds,
    bounds.get_center(),
)


# ---------------------------------------------------------
# 14. Add 3D bridge label
# ---------------------------------------------------------

label = scene_widget.add_3d_label(
    [0.0, 0.0, 0.0],
    bridge_name,
)

label.scale = 1.2


# ---------------------------------------------------------
# 15. Metadata panel
# ---------------------------------------------------------

em = window.theme.font_size

panel = gui.Vert(
    0.5 * em,
    gui.Margins(
        em,
        em,
        em,
        em,
    ),
)


title = gui.Label(
    bridge_name
)

title.text_color = gui.Color(
    1.0,
    1.0,
    1.0,
)

panel.add_child(
    title
)


panel.add_child(
    gui.Label(
        f"NVDB ID: {bridge_id}"
    )
)

panel.add_child(
    gui.Label(
        f"Vegsystem: {road_reference}"
    )
)

panel.add_child(
    gui.Label(
        f"Datafangst: {capture_date}"
    )
)

panel.add_child(
    gui.Label(
        f"EPSG: {epsg_code}"
    )
)

panel.add_child(
    gui.Label(
        f"Punkter i fil: "
        f"{header.point_count:,}"
    )
)

panel.add_child(
    gui.Label(
        f"Punkter vist: "
        f"{sample_size:,}"
    )
)

panel.add_child(
    gui.Label(
        f"Filstørrelse: "
        f"{copc_size / 1024**2:.1f} MB"
    )
)

panel.add_child(
    gui.Label(
        f"Utstrekning X: "
        f"{extent[0]:.1f} m"
    )
)

panel.add_child(
    gui.Label(
        f"Utstrekning Y: "
        f"{extent[1]:.1f} m"
    )
)

panel.add_child(
    gui.Label(
        f"Utstrekning Z: "
        f"{extent[2]:.1f} m"
    )
)

panel.add_child(
    gui.Label(
        "Kilde: Saga"
    )
)


# ---------------------------------------------------------
# 16. Add spacer / info heading
# ---------------------------------------------------------

panel.add_child(
    gui.Label("")
)

panel.add_child(
    gui.Label(
        "Tilstandsdata"
    )
)

panel.add_child(
    gui.Label(
        "Punktsky tilgjengeliggjort "
        "direkte fra Saga"
    )
)


# ---------------------------------------------------------
# 17. Reset camera button
# ---------------------------------------------------------

reset_button = gui.Button(
    "Tilpass visning"
)


def reset_camera():
    scene_widget.setup_camera(
        60,
        bounds,
        bounds.get_center(),
    )


reset_button.set_on_clicked(
    reset_camera
)

panel.add_child(
    reset_button
)


# ---------------------------------------------------------
# 18. Layout
# ---------------------------------------------------------

def on_layout(layout_context):
    rect = window.content_rect

    panel_width = int(
        320
    )

    scene_widget.frame = gui.Rect(
        rect.x,
        rect.y,
        rect.width - panel_width,
        rect.height,
    )

    panel.frame = gui.Rect(
        rect.get_right() - panel_width,
        rect.y,
        panel_width,
        rect.height,
    )


window.set_on_layout(
    on_layout
)


# ---------------------------------------------------------
# 19. Add widgets to window
# ---------------------------------------------------------

window.add_child(
    scene_widget
)

window.add_child(
    panel
)


# ---------------------------------------------------------
# 20. Run GUI
# ---------------------------------------------------------

app.run()