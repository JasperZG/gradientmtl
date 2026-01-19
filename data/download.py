"""Download MoleculeNet datasets from public sources."""

import urllib.request
from pathlib import Path
import pandas as pd


DATASET_URLS = {
    'bace': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
    'bbbp': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
    'esol': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
    'lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
    'herg': 'https://dataverse.harvard.edu/api/access/datafile/4259588',  # TDC hERG
}


def download_file(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a file from URL to local path."""
    try:
        # Add user-agent header for Harvard Dataverse
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_datasets(output_dir: Path, config: dict) -> dict[str, Path]:
    """
    Download all datasets specified in config.

    Args:
        output_dir: Directory to save downloaded files
        config: Configuration dict with task definitions

    Returns:
        Dict mapping task names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    for task_name, task_config in config['tasks'].items():
        # Handle local files (already downloaded)
        if task_config.get('url') == 'local':
            # Look for existing hERG file
            local_path = output_dir / 'herg_raw.tab'
            if local_path.exists():
                downloaded[task_name] = local_path
                print(f"Using local file for {task_name}: {local_path}")
            else:
                # Try to download from TDC
                url = DATASET_URLS.get(task_name)
                if url:
                    output_path = output_dir / f'{task_name}_raw.tab'
                    print(f"Downloading {task_name} from TDC...")
                    if download_file(url, output_path):
                        downloaded[task_name] = output_path
                        print(f"Downloaded {task_name} to {output_path}")
                    else:
                        print(f"Failed to download {task_name}")
            continue

        url = task_config.get('url') or DATASET_URLS.get(task_name)
        if not url:
            print(f"No URL found for task: {task_name}")
            continue

        # Determine output filename
        if url.endswith('.csv'):
            ext = '.csv'
        elif 'dataverse' in url:
            ext = '.tab'
        else:
            ext = '.csv'

        output_path = output_dir / f'{task_name}_raw{ext}'

        # Skip if already exists
        if output_path.exists():
            downloaded[task_name] = output_path
            print(f"Already exists: {output_path}")
            continue

        print(f"Downloading {task_name}...")
        if download_file(url, output_path):
            downloaded[task_name] = output_path
            print(f"Downloaded {task_name} to {output_path}")
        else:
            print(f"Failed to download {task_name}")

    return downloaded


def load_dataset(path: Path, task_config: dict) -> pd.DataFrame:
    """
    Load a dataset file and extract relevant columns.

    Args:
        path: Path to the downloaded file
        task_config: Config dict for this task

    Returns:
        DataFrame with 'smiles' and 'label' columns
    """
    # Determine separator
    if path.suffix == '.tab':
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)

    # Get column names from config
    smiles_col = task_config['smiles_column']
    target_col = task_config['target_column']

    # Validate columns exist
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {path}. "
                        f"Available columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}. "
                        f"Available columns: {list(df.columns)}")

    # Extract and rename
    result = pd.DataFrame({
        'smiles': df[smiles_col],
        'label': df[target_col]
    })

    # Drop rows with missing values
    original_len = len(result)
    result = result.dropna()
    if len(result) < original_len:
        print(f"Dropped {original_len - len(result)} rows with missing values")

    return result


if __name__ == '__main__':
    # Test download
    import yaml

    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(__file__).parent.parent / 'outputs' / 'raw_data'
    downloaded = download_datasets(output_dir, config)

    print(f"\nDownloaded {len(downloaded)} datasets:")
    for name, path in downloaded.items():
        df = load_dataset(path, config['tasks'][name])
        print(f"  {name}: {len(df)} samples")
