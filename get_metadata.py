import io
import os

import boto3
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv


load_dotenv()

endpoint = os.environ["saga_s3_endpoint"]
bucket = os.environ["metadata_bucket"]
access_key = os.environ["metadata_access_key"]
secret_key = os.environ["metadata_secret_key"]

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)


def read_parquet_folder(prefix: str) -> pd.DataFrame:
    paginator = s3.get_paginator("list_objects_v2")
    dataframes = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Skip Delta transaction logs
            if "/_delta_log/" in key:
                continue

            response = s3.get_object(
                Bucket=bucket,
                Key=key,
            )

            content = response["Body"].read()

            # Parquet files start with PAR1
            if not content.startswith(b"PAR1"):
                print(f"Skipping non-Parquet object: {key}")
                continue

            try:
                table = pq.read_table(io.BytesIO(content))
                df = table.to_pandas()

                # Added by us for lineage/debugging
                df["source_key"] = key

                dataframes.append(df)

            except Exception as exc:
                print(f"Failed reading {key}: {exc}")

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


folders = {
    "metadata": "metadata-2ca6bd38c6724640bf8f3342a7a093c2/",
    "stedfesting": "stedfesting-e105671c75834b8b966ecdb85048eb5d/",
    "assosierte_vegobjekter": (
        "assosierte_vegobjekter-08cfd66be1004ffb967206f7f18dc967/"
    ),
    "nabofiler": "nabofiler-05a70cd2605b416eb69c5305b5ac844b/",
}


tables = {}

for name, prefix in folders.items():
    print("\n" + "=" * 80)
    print(f"Reading table: {name}")
    print(f"Prefix: {prefix}")
    print("=" * 80)

    df = read_parquet_folder(prefix)

    tables[name] = df

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    if not df.empty:
        print(df.head(3).to_string())
    else:
        print("No rows found")


metadata_df = tables["metadata"]
stedfesting_df = tables["stedfesting"]
vegobjekter_df = tables["assosierte_vegobjekter"]
nabofiler_df = tables["nabofiler"]


print("\n\nCOMMON COLUMNS BETWEEN TABLES")
print("=" * 80)

names = list(tables.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name1 = names[i]
        name2 = names[j]

        common_columns = (
            set(tables[name1].columns)
            & set(tables[name2].columns)
        )

        print(f"\n{name1} <-> {name2}:")
        print(sorted(common_columns))


# -------------------------------------------------------------------
# Export relational samples
# -------------------------------------------------------------------

output_dir = "samples"
os.makedirs(output_dir, exist_ok=True)

sample_key_count = 10

if (
    not metadata_df.empty
    and "punktsky_versjon_nokkel" in metadata_df.columns
):
    selected_keys = (
        metadata_df["punktsky_versjon_nokkel"]
        .dropna()
        .drop_duplicates()
        .head(sample_key_count)
    )

    print("\n\nSELECTED PUNKTSKY VERSION KEYS")
    print("=" * 80)

    for key in selected_keys:
        print(key)

    print("\n\nWRITING SAMPLE CSV FILES")
    print("=" * 80)

    for name, df in tables.items():
        if df.empty:
            print(f"{name}: skipped because table is empty")
            continue

        if "punktsky_versjon_nokkel" not in df.columns:
            print(
                f"{name}: skipped because "
                "'punktsky_versjon_nokkel' is missing"
            )
            continue

        sample = df[
            df["punktsky_versjon_nokkel"].isin(selected_keys)
        ].copy()

        output_path = os.path.join(
            output_dir,
            f"{name}_sample.csv",
        )

        sample.to_csv(
            output_path,
            index=False,
        )

        print(
            f"{name}: "
            f"{len(sample)} rows -> {output_path}"
        )

else:
    print(
        "\nCould not create relational samples because "
        "metadata is empty or punktsky_versjon_nokkel is missing."
    )