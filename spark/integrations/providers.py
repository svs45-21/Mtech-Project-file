from __future__ import annotations

import os
import requests


class WeatherClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("WEATHER_API_KEY", "")

    def current_weather(self, city: str) -> dict:
        if not self.api_key:
            return {"error": "Missing WEATHER_API_KEY"}
        url = "https://api.openweathermap.org/data/2.5/weather"
        response = requests.get(url, params={"q": city, "appid": self.api_key, "units": "metric"}, timeout=5)
        response.raise_for_status()
        return response.json()
