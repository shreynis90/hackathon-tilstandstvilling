import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

METADATA_FILE = "samples/metadata_sample.csv"
VEGOBJEKTER_FILE = "samples/assosierte_vegobjekter_sample.csv"


# ---------------------------------------------------------
# Saga connection
# ---------------------------------------------------------

endpoint = os.environ["saga_s3_endpoint"]
data_bucket = os.environ["data_bucket"]
access_key = os.environ["data_access_key"]
secret_key = os.environ["data_secret_key"]


s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)


# ---------------------------------------------------------
# Check whether COPC exists
# ---------------------------------------------------------

def object_exists(key: str) -> bool:
    try:
        s3.head_object(
            Bucket=data_bucket,
            Key=key,
        )

        return True

    except ClientError as exc:
        code = exc.response[
            "Error"
        ].get(
            "Code"
        )

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise


# ---------------------------------------------------------
# Load bridge data
# ---------------------------------------------------------

def load_bridge_data() -> pd.DataFrame:

    metadata = pd.read_csv(
        METADATA_FILE
    )

    vegobjekter = pd.read_csv(
        VEGOBJEKTER_FILE
    )


    # -----------------------------------------------------
    # Point clouds marked as bridge
    # -----------------------------------------------------

    bridge_metadata = metadata[
        metadata["bru_flagg"] == True
    ].copy()


    # -----------------------------------------------------
    # Associated NVDB bridge objects
    # -----------------------------------------------------

    bridge_objects = vegobjekter[
        vegobjekter["vegobjekt_type"] == "Bru"
    ].copy()


    # -----------------------------------------------------
    # Join
    # -----------------------------------------------------

    bridges = bridge_metadata.merge(
        bridge_objects,
        on="punktsky_versjon_nokkel",
        how="inner",
        suffixes=(
            "_metadata",
            "_vegobjekt",
        ),
    )


    wanted_columns = [
        "punktsky_nokkel",
        "punktsky_versjon_nokkel",
        "datafangstdato",
        "vegsystemreferanse",
        "vegobjekt_id",
        "vegobjekt_navn",
        "vedlikeholdsansvarlig",
        "copc_filnavn",
        "antall_punkter",
        "epsg_kode",
        "bru_flagg",
        "tunnel_flagg",
    ]


    available_columns = [
        column
        for column in wanted_columns
        if column in bridges.columns
    ]


    bridges = (
        bridges[
            available_columns
        ]
        .drop_duplicates()
        .copy()
    )


    return bridges


# ---------------------------------------------------------
# Cache availability checks
# ---------------------------------------------------------

_copc_availability_cache = {}


def copc_available(
    key: str,
) -> bool:

    key = str(
        key
    )


    if key not in _copc_availability_cache:

        _copc_availability_cache[
            key
        ] = object_exists(
            key
        )


    return _copc_availability_cache[
        key
    ]


# ---------------------------------------------------------
# Available bridge data
# ---------------------------------------------------------

def get_available_bridge_data():

    bridges = load_bridge_data()


    bridges[
        "copc_available"
    ] = bridges[
        "copc_filnavn"
    ].apply(
        copc_available
    )


    bridges = bridges[
        bridges[
            "copc_available"
        ]
    ].copy()


    return bridges


# ---------------------------------------------------------
# API-friendly bridge list
# ---------------------------------------------------------

def get_bridges():

    bridges = (
        get_available_bridge_data()
    )


    bridge_list = (
        bridges[
            [
                "vegobjekt_id",
                "vegobjekt_navn",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "vegobjekt_navn"
        )
        .reset_index(
            drop=True
        )
    )


    result = []


    for _, row in bridge_list.iterrows():

        bridge_id = row[
            "vegobjekt_id"
        ]

        captures = bridges[
            bridges[
                "vegobjekt_id"
            ]
            == bridge_id
        ]


        result.append(
            {
                "bridge_id": str(
                    bridge_id
                ),
                "bridge_name": str(
                    row[
                        "vegobjekt_navn"
                    ]
                ),
                "capture_count": int(
                    captures[
                        "datafangstdato"
                    ]
                    .dropna()
                    .nunique()
                ),
            }
        )


    return result


# ---------------------------------------------------------
# Captures for one bridge
# ---------------------------------------------------------

def get_bridge_captures(
    bridge_id,
):

    bridges = (
        get_available_bridge_data()
    )


    bridge_id_string = str(
        bridge_id
    )


    bridge_rows = bridges[
        bridges[
            "vegobjekt_id"
        ]
        .astype(str)
        == bridge_id_string
    ].copy()


    bridge_rows = (
        bridge_rows
        .sort_values(
            [
                "datafangstdato",
                "copc_filnavn",
            ]
        )
        .drop_duplicates(
            subset=[
                "punktsky_versjon_nokkel",
                "copc_filnavn",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    captures = []


    for index, row in bridge_rows.iterrows():

        captures.append(
            {
                "capture_id": index,
                "bridge_id": str(
                    row[
                        "vegobjekt_id"
                    ]
                ),
                "bridge_name": str(
                    row[
                        "vegobjekt_navn"
                    ]
                ),
                "capture_date": str(
                    row.get(
                        "datafangstdato",
                        "",
                    )
                ),
                "road_reference": str(
                    row.get(
                        "vegsystemreferanse",
                        "",
                    )
                ),
                "copc_key": str(
                    row[
                        "copc_filnavn"
                    ]
                ),
                "epsg": str(
                    row.get(
                        "epsg_kode",
                        "",
                    )
                ),
                "point_count": str(
                    row.get(
                        "antall_punkter",
                        "",
                    )
                ),
                "maintenance_owner": str(
                    row.get(
                        "vedlikeholdsansvarlig",
                        "",
                    )
                ),
            }
        )


    return captures