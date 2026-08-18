import io
import json
from collections import defaultdict
from pathlib import Path

import requests
import reverse_geocode
from PIL import Image

downloadedImgs = 0

# apply config
with open("config.json", "r") as file:
    config = json.load(file)
    numLocations = config["numLocations"]
    locationsFilePath = config["locationsFilePath"]
    outputDirectory = Path(config["outputDirectory"])
    trainSplit = config["trainSplit"]
    validationSplit = config["validationSplit"]

if (trainSplit < 0) or (validationSplit < 0) or (trainSplit + validationSplit >= 1):
    raise ValueError("trainSplit and validationSplit must add up to less than 1")


# returns the country code from input coordinates
def country_bounds(lat, lon):
    coordinates = [(lat, lon)]
    result = reverse_geocode.search(coordinates)
    result = result[0]['country_code'].lower()
    if result == 'gb': # return uk instead of gb bc i already named my dataset folders and i was lazy
        result = 'uk'

    return result


def download_and_crop(url, name):
    # images are usually squares but the bottom half is all black so cut that out
    crop_box = (0, 0, 512, 256) # (left, upper, right, lower)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # raises an error for bad responses

        # open the image
        image_bytes = io.BytesIO(response.content)
        img = Image.open(image_bytes)

        # scale from 512x256 to 320x160
        cropped_img = img.crop(crop_box)
        cropped_img = cropped_img.resize((320, 160))

        # save image
        cropped_img.save(str(name) + ".jpg")
        print(str(name) + " downloaded, cropped, and saved successfully")

        downloadedImgs += 1

    except requests.exceptions.RequestException as e:
        print(f"network error occurred: {e}")
    except Exception as e:
        print(f"an error occurred: {e}")


with open(locationsFilePath, "r") as file:
    locations = json.load(file)

availableLocations = locations['customCoordinates']

if numLocations > len(availableLocations):
    raise ValueError(f"numLocations is {numLocations} but the file only has {len(availableLocations)} locations")

# keeps each country's split balanced even if the location file is not sorted
countryCounts = defaultdict(int)

# add every image to proper folder
for i in range(numLocations):
    lat = availableLocations[i]["lat"]
    lon = availableLocations[i]["lng"]
    country = country_bounds(lat, lon)
    countryIndex = countryCounts[country]
    countryCounts[country] += 1

    # use a repeating group of 100 for an even split within each country
    splitPosition = countryIndex % 100
    if splitPosition < trainSplit * 100: # 80% training
        split = "train"
    elif splitPosition < (trainSplit + validationSplit) * 100: # 10% validation
        split = "validation"
    else:                                                     # 10% testing
        split = "test"

    outputPath = outputDirectory / split / country
    outputPath.mkdir(parents=True, exist_ok=True)

    url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid=" + availableLocations[i]['panoId'] + "&output=tile&x=0&y=0&zoom=0&nbt=1&fover=2"
    imgName = outputPath / f"{country}-{countryIndex}"

    download_and_crop(url, imgName)

print(f"successfully processed {downloadedImgs}/{numLocations} images")
