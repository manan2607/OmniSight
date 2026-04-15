import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ThreadPoolExecutor
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import sqlite3
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from pillow_heif import register_heif_opener

register_heif_opener()


# GeoLocator Config
geolocator = Nominatim(user_agent="geo_app_2026")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=3)

geo_cache = {}



# BLIP Config
device = "mps" if torch.backends.mps.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    use_fast=True
)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

blip_model = blip_model.to(device)
blip_model.eval()






def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=30)

    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption



def exact_geo(lat, lon):
    key = (round(lat, 4), round(lon, 4))  

    if key in geo_cache:
        return geo_cache[key]
     
    for i in range(2):
        try:
            location = reverse(f"{lat}, {lon}")
            address = location.address if location else None
        except:
            address = None

        if address:
            break

    geo_cache[key] = address
    return address


def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def get_exact_details(gps_info):
    try:
        lat = convert_to_degrees(gps_info[2])
        if gps_info[1] != 'N':
            lat = -lat

        lon = convert_to_degrees(gps_info[4])
        if gps_info[3] != 'E':
            lon = -lon

        return lat, lon
    except:
        return None, None
    

def get_processed_files():
    conn = sqlite3.connect("photo_metadata.db")
    cursor = conn.cursor()

    cursor.execute("SELECT file_path FROM metadata")
    rows = cursor.fetchall()

    conn.close()

    return set(r[0] for r in rows)

def process_image(file_path):
    processed_files = get_processed_files()

    try:
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}
        if file_path.suffix.lower() not in extensions:
            return None

        if file_path.name in processed_files:
            return None

        img = Image.open(file_path)

        if file_path.suffix.lower() == '.heic':
            exif_data = img.getexif()
            gps_info = exif_data.get_ifd(0x8825)
        else:
            exif_data = img._getexif()
            gps_info = exif_data.get(34853) if exif_data else None

        lat, lon = None, None
        address = None

        if gps_info:
            lat, lon = get_exact_details(gps_info)
            if lat and lon:
                address = exact_geo(lat, lon)

        datetime = exif_data.get(306) if exif_data else None


        #BLIP
        caption = generate_caption(img)

        return (
            str(Path(file_path).name),
            str(datetime),
            str(address),
            str(caption)
        )

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def meta_data_pipeline(image_paths):
    conn = sqlite3.connect('photo_metadata.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        date_time TEXT,
        location TEXT,
        caption TEXT,
        processed INTEGER DEFAULT 0
    )
    ''')

    conn.commit()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_image, image_paths))

    results = [r for r in results if r is not None]
    cursor.executemany('''
        INSERT OR REPLACE INTO metadata (file_path, date_time, location, caption)
        VALUES (?, ?, ?, ?)
    ''', results)

    conn.commit()
    conn.close()
 
    print(f"✅ Processed {len(results)} images successfully!")