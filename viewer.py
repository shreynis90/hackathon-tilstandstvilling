import os
import threading
import webbrowser
from functools import lru_cache

import boto3
import laspy
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pyproj import Transformer

from select_case import (
    get_bridges,
    get_bridge_captures,
)


load_dotenv()


# ---------------------------------------------------------
# Configuration
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


POINT_SAMPLE_SIZE = 25_000


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="TilstandsTvilling"
)


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

@app.get("/")
def index():

    return FileResponse(
        "static/index.html"
    )


# ---------------------------------------------------------
# Bridges
# ---------------------------------------------------------

@app.get(
    "/api/bridges"
)
def bridges():

    return get_bridges()


# ---------------------------------------------------------
# Captures for bridge
# ---------------------------------------------------------

@app.get(
    "/api/bridges/{bridge_id}/captures"
)
def bridge_captures(
    bridge_id: str,
):

    captures = get_bridge_captures(
        bridge_id
    )


    if not captures:

        raise HTTPException(
            status_code=404,
            detail=(
                "No available captures "
                "for this bridge."
            ),
        )


    return captures


# ---------------------------------------------------------
# Find capture
# ---------------------------------------------------------

def find_capture(
    bridge_id: str,
    capture_id: int,
):

    captures = get_bridge_captures(
        bridge_id
    )


    for capture in captures:

        if (
            capture[
                "capture_id"
            ]
            == capture_id
        ):

            return capture


    raise HTTPException(
        status_code=404,
        detail="Capture not found.",
    )


# ---------------------------------------------------------
# Download COPC
# ---------------------------------------------------------

def download_copc(
    copc_key: str,
):

    os.makedirs(
        "data",
        exist_ok=True,
    )


    local_path = os.path.join(
        "data",
        os.path.basename(
            copc_key
        ),
    )


    if os.path.exists(
        local_path
    ):

        print(
            "Using cached file:",
            local_path,
        )

        return local_path


    print(
        "Downloading:",
        copc_key,
    )


    s3.download_file(
        bucket,
        copc_key,
        local_path,
    )


    print(
        "Downloaded:",
        local_path,
    )


    return local_path


# ---------------------------------------------------------
# Read and prepare point cloud
# ---------------------------------------------------------

@lru_cache(
    maxsize=4
)
def prepare_point_cloud(
    copc_key: str,
    epsg_code: str,
):

    print(
        "\nPreparing point cloud:"
    )

    print(
        copc_key
    )


    # -----------------------------------------------------
    # Object metadata
    # -----------------------------------------------------

    response = s3.head_object(
        Bucket=bucket,
        Key=copc_key,
    )


    file_size = response[
        "ContentLength"
    ]


    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    local_path = download_copc(
        copc_key
    )


    # -----------------------------------------------------
    # LAS / COPC
    # -----------------------------------------------------

    print(
        "Reading point cloud..."
    )


    las = laspy.read(
        local_path
    )


    total_points = len(
        las.points
    )


    sample_size = min(
        POINT_SAMPLE_SIZE,
        total_points,
    )


    # -----------------------------------------------------
    # Sample
    # -----------------------------------------------------

    rng = np.random.default_rng(
        seed=42
    )


    indices = rng.choice(
        total_points,
        size=sample_size,
        replace=False,
    )


    x = np.asarray(
        las.x
    )[indices]

    y = np.asarray(
        las.y
    )[indices]

    z = np.asarray(
        las.z
    )[indices]


    # -----------------------------------------------------
    # CRS
    # -----------------------------------------------------

    epsg_string = str(
        epsg_code
    ).strip()


    if epsg_string.lower().startswith(
        "epsg:"
    ):

        source_crs = (
            epsg_string
        )

    else:

        source_crs = (
            f"EPSG:{epsg_string}"
        )


    transformer = (
        Transformer.from_crs(
            source_crs,
            "EPSG:4326",
            always_xy=True,
        )
    )


    lon, lat = (
        transformer.transform(
            x,
            y,
        )
    )


    # -----------------------------------------------------
    # RGB
    # -----------------------------------------------------

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

        red = np.asarray(
            las.red
        )[indices]

        green = np.asarray(
            las.green
        )[indices]

        blue = np.asarray(
            las.blue
        )[indices]


        max_rgb = max(
            red.max(),
            green.max(),
            blue.max(),
        )


        if max_rgb > 255:

            red = (
                red / 256
            ).astype(
                np.uint8
            )

            green = (
                green / 256
            ).astype(
                np.uint8
            )

            blue = (
                blue / 256
            ).astype(
                np.uint8
            )


    else:

        red = np.full(
            sample_size,
            60,
            dtype=np.uint8,
        )

        green = np.full(
            sample_size,
            180,
            dtype=np.uint8,
        )

        blue = np.full(
            sample_size,
            220,
            dtype=np.uint8,
        )


    # -----------------------------------------------------
    # Build browser payload
    # -----------------------------------------------------

    points = [
        {
            "position": [
                float(
                    lon[i]
                ),
                float(
                    lat[i]
                ),
                float(
                    z[i]
                ),
            ],
            "color": [
                int(
                    red[i]
                ),
                int(
                    green[i]
                ),
                int(
                    blue[i]
                ),
            ],
        }
        for i in range(
            sample_size
        )
    ]


    center = {
        "longitude": float(
            np.mean(
                lon
            )
        ),
        "latitude": float(
            np.mean(
                lat
            )
        ),
        "height": float(
            np.mean(
                z
            )
        ),
    }


    print(
        f"Prepared "
        f"{sample_size:,} points."
    )


    return {
        "points": points,
        "total_points": int(
            total_points
        ),
        "displayed_points": int(
            sample_size
        ),
        "file_size_mb": round(
            file_size
            / 1024**2,
            1,
        ),
        "center": center,
        "source_crs": source_crs,
    }


# ---------------------------------------------------------
# Case metadata
# ---------------------------------------------------------

@app.get(
    "/api/bridges/{bridge_id}/captures/{capture_id}"
)
def capture_metadata(
    bridge_id: str,
    capture_id: int,
):

    capture = find_capture(
        bridge_id,
        capture_id,
    )


    cloud = prepare_point_cloud(
        capture[
            "copc_key"
        ],
        capture[
            "epsg"
        ],
    )


    return {
        "asset_name": capture[
            "bridge_name"
        ],
        "asset_id": capture[
            "bridge_id"
        ],
        "capture_date": capture[
            "capture_date"
        ],
        "road_reference": capture[
            "road_reference"
        ],
        "maintenance_owner": capture[
            "maintenance_owner"
        ],
        "epsg": cloud[
            "source_crs"
        ],
        "source": "Saga",
        "copc_key": capture[
            "copc_key"
        ],
        "file_size_mb": cloud[
            "file_size_mb"
        ],
        "total_points": cloud[
            "total_points"
        ],
        "displayed_points": cloud[
            "displayed_points"
        ],
        "center": cloud[
            "center"
        ],
    }


# ---------------------------------------------------------
# Point-cloud endpoint
# ---------------------------------------------------------

@app.get(
    "/api/bridges/{bridge_id}/captures/{capture_id}/points"
)
def capture_points(
    bridge_id: str,
    capture_id: int,
):

    capture = find_capture(
        bridge_id,
        capture_id,
    )


    cloud = prepare_point_cloud(
        capture[
            "copc_key"
        ],
        capture[
            "epsg"
        ],
    )


    return {
        "points": cloud[
            "points"
        ]
    }


# ---------------------------------------------------------
# Open browser
# ---------------------------------------------------------

def open_browser():

    webbrowser.open(
        "http://127.0.0.1:8000"
    )


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

if __name__ == "__main__":

    threading.Timer(
        1.5,
        open_browser,
    ).start()


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )