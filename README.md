# Project Title

A brief description of what this project does and who it's for.

## Table of Contents

1. [About](#about)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)

## About

A foundation models based CLAM ensemble to predict TFE3-rearraged renal cell carcinoma based on H&E-stained slide images.
## Getting Started

### Prerequisites
For environment prerequisites, please refer to the original repos:
- UNI (https://github.com/mahmoodlab/uni)
- CHIEF (https://github.com/hms-dbmi/CHIEF)
- CONCH (https://github.com/mahmoodlab/CONCH)
- Prov-GigaPath (https://huggingface.co/prov-gigapath/prov-gigapath)
- Virchow (https://huggingface.co/paige-ai/Virchow)

## Usage
1. Patch generation using slides and corresponding annotations. See details at `create_patches_custom.py`
2. Color Normalization using staintools with Vahadane method. Install staintools via `pip install staintools`. See details at `color_normalization.py`.
3. Feature extraction using `Modified_CLAM/extract_features.py`.
4. Datasplit using `Modified_CLAM/create_splits_seq.py`.
5. CLAM training using `Modified_CLAM/main.py`.
6. Evaluation using `Modified_CLAM/eval.py`.
7. For attention heatmap generation
   - download example slide, which is available at https://drive.google.com/file/d/199TtnQQxTrAVCmSfaQWvmjlMZSbDsv1P/view?usp=drive_link.
The example slide with corresponding CHIEF and UNI checkpoints used for attention heatmap generation is provided to reproduce the partial result.
1. Download the example slide, which is available at https://drive.google.com/file/d/199TtnQQxTrAVCmSfaQWvmjlMZSbDsv1P/view?usp=drive_link.
2. 
- Example:
  ```bash
  python main.py
  ```
- Mention any configurations or additional steps required to run the project.

## Contributing

If you want others to contribute, outline the guidelines here.

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request.

## License

State the license type (e.g., MIT) and provide a link to the full license.

- Example:
  ```
  MIT License. See [LICENSE](LICENSE) for details.
  ```

## Contact

Provide your contact details or links where users can reach you for more information.

- GitHub: [@your_username](https://github.com/your_username)
- Email: your_email@example.com
