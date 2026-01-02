"""Weather-related tools for agents."""

from langchain_core.tools import tool
from typing import List


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location.
    
    This tool provides real-time weather information including:
    - Current temperature
    - Weather conditions (sunny, cloudy, rainy, etc.)
    - General atmospheric conditions
    
    Use this tool whenever you need to provide current weather data
    to users. The location should be as specific as possible for
    accurate results.
    
    Args:
        location: The city and state/country to get weather for.
                 Format: "City, State/Country" (e.g., "San Francisco, CA")
    
    Returns:
        str: A description of the current weather conditions including
             temperature and general conditions.
    
    Example:
        >>> get_weather("London, UK")
        "The weather in London, UK is cloudy and 15°C with light rain."
    
    Note:
        In production, this should integrate with a real weather API
        such as OpenWeatherMap, WeatherAPI.com, or similar service.
    """
    # TODO: Replace with actual weather API integration
    # Example APIs:
    # - OpenWeatherMap: https://openweathermap.org/api
    # - WeatherAPI: https://www.weatherapi.com/
    # - NOAA: https://www.weather.gov/documentation/services-web-api
    
    return f"The weather in {location} is sunny and 72°F with clear skies."


# Registry of all weather-related tools
WEATHER_TOOLS: List = [get_weather]