import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_raw_volume(path: Path) -> np.ndarray:
    data: nib.Nifti1Image = nib.load(path)
    data = nib.as_closest_canonical(data)  # set scan axis orientation
    raw = data.get_fdata(caching='unchanged', dtype=np.float32)
    return raw


def load_labels_volume(path: Path) -> np.ndarray:
    labels = load_raw_volume(path).astype(np.uint8)
    labels[labels > 1] = 0  # ignore labels higher than 1
    return labels


database = Path('./WMH_database')
slices = Path('./Slices')
if not database.is_dir():
    #TODO add wget from cloud database, create WMH_database directory
    pass

for brain in tqdm(database.iterdir(), desc="Extracting brain slices", total=len(list(database.iterdir()))):
    for scan in brain.glob('*.nii.gz'):
        if 'labels_wmh' in str(scan):
            volume = load_labels_volume(scan)
        else:
            volume = load_raw_volume(scan)
        for z in range(volume.shape[2]):
            filepath = Path(slices, scan.stem.split('.')[0])
            filepath.mkdir(parents=True, exist_ok=True)
            filename = scan.parent.stem + '_' + str(z) + '.png'
            plt.imsave(filepath / filename, volume[:, :, z], cmap="gray", origin="lower")
