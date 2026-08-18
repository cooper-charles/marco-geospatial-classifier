import io
import requests
from PIL import Image
import json
import reverse_geocode

with open("config.json", "r") as file:
    config = json.load(file)
    numLocations = config["numLocations"]
    numCountries = config["numCountries"]
    locationsFilePath = config["locationsFilePath"]

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
        cropped_img.save(name + ".jpg")
        print(name + " downloaded, cropped, and saved successfully")

    except requests.exceptions.RequestException as e:
        print(f"network error occurred: {e}")
    except Exception as e:
        print(f"an error occurred: {e}")


with open(locationsFilePath, "r") as file:
    locations = json.load(file)

# add every image to proper folder
for i in range(numLocations):
    lat = locations['customCoordinates'][i]["lat"]
    lon = locations['customCoordinates'][i]["lng"]
    country = country_bounds(lat, lon)
    url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid=" + locations['customCoordinates'][i]['panoId'] + "&output=tile&x=0&y=0&zoom=0&nbt=1&fover=2"
    if i%(numLocations/numCountries) < (numLocations*.8): # 80% training
        img_name = "D:/dataset4/train/" + country + '/' + country + "-" + str(i%(numLocations/numCountries))
    elif i%(numLocations/numCountries) < (numLocations*.9): # 10% validation
        img_name = "D:/dataset4/validation/" + country + '/' + country + "-" + str(i%(numLocations/numCountries))
    else:                                                    # 10% testing
        img_name = "D:/dataset4/test/" + country + '/' + country + "-" + str(i%(numLocations/numCountries))
    
    download_and_crop(url, img_name)

print(f"Processed {len(locations['customCoordinates'])} images")