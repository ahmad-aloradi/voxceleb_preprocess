from pathlib import Path
from typing import Union, Optional
import pandas as pd
import os
import logging
from voxceleb_prep import VoxCelebProcessor


# Configure the logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Create console handler with a custom format
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
if not log.handlers:
    log.addHandler(ch)


def print_test_statistics(df: pd.DataFrame) -> None:
    log.info(f"Total trials: {len(df)}")
    log.info(f"Positive trials: {(df['label'] == 1).sum()}")
    log.info(f"Same gender trials: {df['same_gender'].sum()}")
    log.info(f"Same nationality trials: {df['same_nationality'].sum()}")


def enrich_verification_file(
    veri_test_path: Union[str, Path],
    metadata_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sep: str = '|'
) -> pd.DataFrame:
    """
    Enrich verification test file with metadata information.
    
    Args:
        veri_test_path: Path to veri_test.txt
        metadata_path: Path to vox_metadata.csv
        output_path: Optional path to save the enriched CSV
        
    Returns:
        DataFrame with enriched verification trials
    """
    # Read metadata
    metadata_df = pd.read_csv(metadata_path, sep='|').fillna('N/A')

    # Create lookup dictionaries for faster access
    metadata_lookup = metadata_df.set_index('speaker_id').to_dict('index')

    # Read verification file
    veri_df = pd.read_csv(veri_test_path, sep=' ', header=None,
                        names=['label', 'enroll_path', 'test_path'])

    # Extract speaker IDs from paths
    veri_df['enroll_id'] = veri_df['enroll_path'].apply(lambda x: x.split('/')[0])
    veri_df['test_id'] = veri_df['test_path'].apply(lambda x: x.split('/')[0])
    
    # Add metadata for both speakers
    for field in ['nationality', 'gender', 'class_id']:
        veri_df[f'enroll_{field}'] = veri_df['enroll_id'].map(
            lambda x: metadata_lookup[x][field] if x in metadata_lookup else 'N/A'
        )
        veri_df[f'test_{field}'] = veri_df['test_id'].map(
            lambda x: metadata_lookup[x][field] if x in metadata_lookup else 'N/A'
        )
    
    # Add trial type (same/different nationality, gender)
    veri_df['same_nationality'] = (
        (veri_df['enroll_nationality'] == veri_df['test_nationality']) & 
        (veri_df['enroll_nationality'] != 'N/A')
    ).astype(int)

    veri_df['same_gender'] = (
        veri_df['enroll_gender'] == veri_df['test_gender']
    ).astype(int)

    # Reorder columns for clarity
    column_order = [
        'label', 
        'enroll_path', 'test_path',
        'enroll_id', 'test_id',
        'enroll_gender', 'test_gender',
        'enroll_nationality', 'test_nationality',
        'enroll_class_id', 'test_class_id',
        'same_gender', 'same_nationality'
    ]
    veri_df = veri_df[column_order]

    # Print statistics
    print_test_statistics(veri_df)

    # Save if output path provided
    if output_path:
        VoxCelebProcessor.save_csv(veri_df, output_path, sep=sep)            
    else:
        return veri_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VoxCeleb metadata")
    parser.add_argument("--root_dir", 
                        type=str,
                        default="data/voxceleb",
                        help="Root directory containing both VoxCeleb1 and VoxCeleb2")
    parser.add_argument("--artifacts_dir", 
                    type=str,
                    default="data/voxceleb/voxceleb_metadata/preprocessed",
                    help="Root directory containing both VoxCeleb1 and VoxCeleb2")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print verbose output")
    parser.add_argument("--preprocess_stats_file", 
                        type=str,
                        default="data/voxceleb/voxceleb_metadata/preprocess_metadata/preprocess_stats.csv")
    parser.add_argument("--min_duration",
                        type=float,
                        default=0.5, 
                        help="Minimum duration in seconds. Utterances shorter than this will be excluded")
    parser.add_argument("--sep", 
                        type=str,
                        default="|",
                        help="Separator used for the metadata file")
    
    args = parser.parse_args()
    
     # Run veri_test.txt enricher
    output_path = 'data/voxceleb/voxceleb_metadata/preprocessed/veri_test_rich.csv' 
    if os.path.exists(output_path):
        log.info(f"Output file already exists: {output_path}")
        enriched_df = pd.read_csv(output_path, sep='|')

    enriched_df = enrich_verification_file(
        'data/voxceleb/voxceleb_metadata/downloaded/veri_test.txt',
        'data/voxceleb/voxceleb_metadata/preprocessed/vox_meta.csv',
        output_path=None
        )
