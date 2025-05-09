from typing import Dict, List, Tuple

class RestaurantFeatures:
    def __init__(self, restaurant_id: str, name: str, url: str,
                 picture_url: str, star_rating: float, description: str,
                 latitude: float, longitude: float, min_price: float,
                 max_price: float, price_level: str, number_of_tables: int,
                 work_times: Dict, features: List[str], tags: List[str],
                 cuisines: List[str], meal_types: List[str],
                 dishes_with_prices: Dict[str, float]):
        self.restaurant_id = restaurant_id
        self.name = name
        self.url = url
        self.picture_url = picture_url
        self.star_rating = star_rating
        self.description = description
        self.latitude = latitude
        self.longitude = longitude
        self.min_price = min_price
        self.max_price = max_price
        self.price_level = price_level
        self.number_of_tables = number_of_tables
        self.work_times = work_times
        self.features = features
        self.tags = tags
        self.cuisines = cuisines
        self.meal_types = meal_types
        self.dishes_with_prices = dishes_with_prices

    def to_dict(self) -> Dict:
        return {
            'restaurant_id': self.restaurant_id,
            'star_rating': self.star_rating,
            'description': self.description,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'price_level': self.price_level,
            'cuisines': self.cuisines,
            'meal_types': self.meal_types,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'features': self.features,
            'tags': self.tags,
            'dishes_with_prices': self.dishes_with_prices
        }