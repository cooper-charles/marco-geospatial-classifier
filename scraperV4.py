import io
import requests
from PIL import Image
import json
import reverse_geocode


def country_bounds(lat, lon):
    coordinates = [(lat, lon)]
    result = reverse_geocode.search(coordinates)
    result = result[0]['country_code'].lower()
    if result == 'gb':
        result = 'uk'

    return result

def download_and_crop(url, name):
    # Box coordinates: (left, upper, right, lower)
    crop_box = (0, 0, 512, 256) 

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # raises an error for bad responses (404, 500, etc.)

        # open the image directly from the byte stream
        image_bytes = io.BytesIO(response.content)
        img = Image.open(image_bytes)

        cropped_img = img.crop(crop_box)
        cropped_img = cropped_img.resize((320, 160))

        cropped_img.save(name + ".jpg")
        print(name + " downloaded, cropped, and saved successfully")

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


with open("locationsV4.json", "r") as file:
    locations = json.load(file)

for i in range(400000):
    lat = locations['customCoordinates'][i]["lat"]
    lon = locations['customCoordinates'][i]["lng"]
    country = country_bounds(lat, lon)
    url = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid=" + locations['customCoordinates'][i]['panoId'] + "&output=tile&x=0&y=0&zoom=0&nbt=1&fover=2"
    if i%20000 < 16000:
        img_name = "D:/dataset4/train/" + country + '/' + country + "-" + str(i%20000)
    elif i%20000 < 18000:
        img_name = "D:/dataset4/validation/" + country + '/' + country + "-" + str(i%20000)
    else:
        img_name = "D:/dataset4/test/" + country + '/' + country + "-" + str(i%20000)
    
    download_and_crop(url, img_name)

print(len(locations['customCoordinates']))