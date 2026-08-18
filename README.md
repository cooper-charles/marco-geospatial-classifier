# GeoGuessr AI

GeoGuessr AI is a computer vision project that predicts the country shown in a Google Street View panorama. I built the data collection pipeline and trained a ConvNeXt-Tiny image classifier on the collected images.

The latest dataset used ten country classes: Australia, Brazil, Italy, Japan, Kenya, Mexico, South Africa, Thailand, the United Kingdom, and the United States.

## How it works

1. Panorama locations are labeled by country using their latitude and longitude.
2. A Street View tile is downloaded, cropped, and resized to `320 x 160`.
3. The images are split into training, validation, and test sets.
4. A pretrained ConvNeXt-Tiny model is fine-tuned with class-weighted cross-entropy loss.
5. The model with the best validation accuracy is evaluated on a held-out test set and saved.

The training pipeline also uses color jitter and a random horizontal panorama roll. Rolling the panorama changes the position of its seam without mirroring geographic clues such as which side of the road traffic uses.

## Project structure

```text
.
|-- train.py         # training and evaluation pipeline
|-- scraper.py       # dataset collection script
|-- config.json      # scraper and training settings
|-- requirements.txt
`-- README.md
```

The image datasets, location exports, and trained checkpoints are intentionally excluded from Git because they are large generated artifacts. A trained checkpoint can be attached separately to a GitHub Release.

## Collecting images

`config.json` controls the location file, output folder, number of locations, and dataset split. Add a location file containing a `customCoordinates` list with `panoId`, `lat`, and `lng` values, then run:

```bash
python scraper.py
```

The scraper creates the train, validation, and test folders automatically. The generated `data/` directory is ignored by Git.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Training

The trainer reads the dataset location and training settings from `config.json`. It expects the output created by the scraper:

```text
data/
|-- train/
|   |-- au/
|   |-- br/
|   `-- ...
|-- validation/
`-- test/
```

Then run:

```bash
python train.py
```

The script fine-tunes ConvNeXt-Tiny, saves the checkpoint with the best validation accuracy, and evaluates that model on the test split.

## Notes

- Training uses CUDA automatically when it is available and otherwise runs on CPU.
- Training images and model weights are not committed to this repository.
- Street View availability and request behavior may change over time.
