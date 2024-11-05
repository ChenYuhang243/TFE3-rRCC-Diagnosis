# Enhancing diagnosis of TFE3-rearranged renal cell carcinoma with pathology foundation models on whole-slide images

This project utilizes a foundation model-based CLAM ensemble to predict TFE3-rearranged renal cell carcinoma from whole slide images.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Usage](#usage)
3. [Contact](#contact)

## Getting Started

### Prerequisites

Ensure you meet the environmental prerequisites listed in the original repositories:

- [UNI](https://github.com/mahmoodlab/uni)
- [CHIEF](https://github.com/hms-dbmi/CHIEF)
- [CONCH](https://github.com/mahmoodlab/CONCH)
- [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)
- [Virchow](https://huggingface.co/paige-ai/Virchow)
- [CLAM](https://github.com/mahmoodlab/CLAM/tree/master)

## Usage

1. **Patch Generation**: Generate patches using slides and corresponding annotations. Details are available in `create_patches_custom.py`.
2. **Color Normalization**: Use staintools with the Vahadane method. Install staintools via:
   ```bash
   pip install staintools
   ```
   Details are available in `color_normalization.py`.
   - For faster color normalization, the Macenko method with GPU implementation is recommended. Refer to [stain_normalizer](https://github.com/ChenYuhang243/stain-normalizer) for more details.
3. **Feature Extraction**: Run `Modified_CLAM/extract_features.py`.
   - Grant access via the links provided above before feature extraction.
   - Update the code in `Modified_CLAM/models/builder.py` as suggested.
4. **Data Splitting**: Use `Modified_CLAM/create_splits_seq.py` to split the data.
5. **CLAM Training**: Train the model using `Modified_CLAM/main.py`.
6. **Evaluation**: Evaluate the model using `Modified_CLAM/eval.py`.
7. **Attention Heatmap Generation**:
   - Download an example slide from [Google Drive](https://drive.google.com/file/d/199TtnQQxTrAVCmSfaQWvmjlMZSbDsv1P/view?usp=drive_link), or get the full codebase from [Zenodo](https://zenodo.org/records/13893259?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZmNjYyZWVjLWFiMWYtNGQ3ZS05YTFjLTQ5ZWQxODQzNDVkZCIsImRhdGEiOnt9LCJyYW5kb20iOiI4ZTZkYzViOTJlYjE4NzcyMDYwOWQwNmI3MWUxMjNkMCJ9.bAU4VNctXqtk6v5vJD6j6PncdlFQDdiDnrvA7TkIs1doteK3Z2ZTXWf9OgHoS-xXHKR9VoUnJ9yH9NFraezXgQ).
   - Update `Modified_CLAM/heatmaps/configs/example.yaml` as directed.
   - Run the following command to generate heatmaps:
     ```bash
     CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config example.yaml
     ```
### Reminder

1. For steps 1 and 2, it's recommended to use multiple workers or multiprocessing. Bash scripts can be generated using `generate_sh.py`.
2. For more information on steps 3-8, please refer to the [CLAM repository](https://github.com/mahmoodlab/CLAM/tree/master).

## Contact

- Email: chenyh238@mail2.sysu.edu.cn
