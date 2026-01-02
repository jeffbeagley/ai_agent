from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    '''Get the current weather for a specific location.
    
    This tool provides real-time weather information including temperature,
    conditions, and general forecast for the requested location.
    
    Args:
        location: The city and state/country (e.g., 'San Francisco, CA' or 'London, UK')
    
    Returns:
        str: Weather information including temperature and conditions
    
    Example:
        >>> get_weather("San Francisco, CA")
        "The weather in San Francisco, CA is sunny and 72°F with clear skies."
    '''
    # TODO: Integrate with real weather API (OpenWeatherMap, WeatherAPI, etc.)
    return f"The weather in {location} is sunny and 72°F with clear skies."

# Registry of all available tools
WEATHER_TOOLS = [get_weather]