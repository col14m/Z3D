## Data Preparation for Z3D Inference
Z3D is evaluated on 2 benchmarks: **ScanRefer** and **Nr3D**.

Before starting, download the annotation files and MaskClustering proposals from our [HuggingFace repo](https://huggingface.co/datasets/drozdgk/Z3D-data).

### ScanNet
Both benchmarks require 3D scene data from the ScanNet dataset.

1. Prepare the folder `./data/scannet` following the instructions from the [Zoo3D repo](https://github.com/col14m/Zoo3D/tree/main/data/scannet).
2. Move the `scannet_oof_infos_val.pkl` annotation file to `./data/`.

After preprocessing, the following data structure should be obtained:
```
./data/
    scannet_oof_infos_val.pkl
    scannet/
        posed_images/
            xxxxx.jpg
            xxxxx.png
            xxxxx.txt
            intrinsic.txt
        rec_posed_images/
            xxxxx.jpg
            xxxxx.png
            xxxxx.txt
            intrinsic.txt
        rec_unposed_images/
            xxxxx.jpg
            xxxxx.png
            xxxxx.txt
            intrinsic.txt
        points/
            scene0011_00.bin
            ...
        points_posed/
            scene0011_00.bin
                ...
        points_unposed/
            scene0011_00.bin
            ...
        ...
```

### ScanRefer
1. Extract the `.tar.gz` archives with preprocessed MaskClustering proposals into `./data/scanrefer`.
2. Move the corresponding `.json` annotation files to `./data/scanrefer`.
3. Download preprocessed Mask3D predictions from [here](https://github.com/CurryYuan/ZSVG3D) and extract them into `./data/scanrefer`.

The following data structure should be obtained:
```
./data/
    scanrefer/
        scanrefer_ann.json
        scanrefer_ann_250.json
        mc_proposals_gt/
        mc_proposals_posed/
        mc_proposals_unposed/
        mask3d_preds/
            Mask3d/
                scannet200/
                    scene0011_00.npz
                    scene0011_01.npz
                    ...
```

### Nr3D
Move the corresponding `.json` annotation files to `./data/nr3d`.

The following data structure should be obtained:
```
./data/
    nr3d/
        nr3d_ann.json
        nr3d_ann_250.json
```