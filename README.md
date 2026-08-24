# MARCO - GeoGuessr AI

MARCO is a computer vision project that predicts the country shown in a Google Street View panorama. I built the data collection pipeline and trained a ConvNeXt-Tiny image classifier on the collected images.

The latest dataset I tried used 25 country classes with 222,000 images: United States, Canada, Mexico, Brazil, Argentina, Chile, Colombia, Peru, United Kingdom, France, Germany, Spain, Italy, Poland, South Africa, Kenya, Ghana, Nigeria, Japan, Thailand, Indonesia, Malaysia, Philippines, Australia, New Zealand.

## How it works

1. Panorama locations are assigned to a country using their latitude and longitude.
2. A Street View tile is downloaded, cropped, and resized. I use `320 x 160`.
3. The images are split into training, validation, and test sets.
4. A pretrained ConvNeXt-Tiny model is fine-tuned with class-weighted cross-entropy loss.
5. The model with the best validation accuracy is evaluated on a held-out test set and saved.

The training pipeline also uses color jitter and a random horizontal panorama roll. Rolling the panorama changes the position of its seam without flipping geographic clues such as the driving side of the road.

## Project structure

```
train.py         # training and evaluation
scraper.py       # dataset collection script
config.json      # scraper and training settings
requirements.txt
README.md
```

I excluded any datasets I used because they can be massive and I didn't want to upload it so you'll have to use the `scraper.py` to create your own.

## Collecting images

`config.json` controls the location file path, output folder path, number of locations, and dataset split. To create a `locations.json` file, I used this [Geoguessr map creator](https://map-g3nerator.vercel.app/). The location file should contain a `customCoordinates` list with `panoId`, `lat`, and `lng` values, then run:

```bash
python scraper.py
```

The scraper creates the train, validation, and test folders automatically.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv

# windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Training

The trainer reads the dataset location and training settings from `config.json`. It expects the output created by the scraper:

```
data/
|-- train/
|   |-- au/
|   |-- br/
|   `-- ...
|-- validation/
`-- test/
```

Then run:

```
python train.py
```

The script fine-tunes ConvNeXt-Tiny, saves the checkpoint with the best validation accuracy, and evaluates that model on the test split.

## Notes

- Training uses CUDA automatically when it is available and otherwise runs on CPU.
- Please be kind to both the map making website and the Google street view API