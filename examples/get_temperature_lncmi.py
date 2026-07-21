import requests


def get_current_weather():
    try:
        # Step 1: Get your current latitude and longitude using your IP address
        geo_url = "https://ipapi.co/json/"
        geo_response = requests.get(geo_url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        
        # Extract location data
        lat = geo_response.get("latitude")
        lon = geo_response.get("longitude")
        city = geo_response.get("city")
        region = geo_response.get("region")
        
        if not lat or not lon:
            print("Could not determine your current location automatically.")
            return

        print(f"Detected Location: {city}, {region} ({lat}, {lon})")

        # Step 2: Query the Open-Meteo API with your coordinates
        # 'temperature_2m' fetches the air temperature 2 meters above the ground
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m",
            "temperature_unit": "celsius"  # Change to "fahrenheit" if preferred
        }
        
        weather_response = requests.get(weather_url, params=params).json()
        
        # Step 3: Extract the current temperature and units
        current_data = weather_response.get("current")
        current_units = weather_response.get("current_units")
        
        temperature = current_data.get("temperature_2m")
        unit = current_units.get("temperature_2m")
        
        print(f"Current Temperature: {temperature}{unit}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")

if __name__ == "__main__":
    get_current_weather()
