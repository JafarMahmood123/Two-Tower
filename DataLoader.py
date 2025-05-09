from UserFeatures import UserFeatures  # Import the class, not the module
from RestaurantFeatures import RestaurantFeatures
import pandas as pd


def load_data_from_csv(user_csv_path, restaurant_csv_path, interactions_csv_path):
    # Load users and rename columns to match class parameters
    users_df = pd.read_csv(user_csv_path)
    users_df = users_df.rename(columns={
        'Id': 'user_id',  # Map 'Id' column to 'user_id'
        'FirstName': 'first_name',
        'LastName': 'last_name',
        'BirthDate': 'birthdate',
        'Email': 'email',
        'Password': 'password',
    })

    users = [
    UserFeatures(**{
        k: v for k, v in row.to_dict().items()
        if k not in ['Age', 'Role']  # Explicitly exclude 'Age' and 'Role'
    })
    for _, row in users_df.iterrows()]

    # Load restaurants
    restaurants_df = pd.read_csv(restaurant_csv_path)
    restaurants = []
    for _, row in restaurants_df.iterrows():
        restaurants.append(RestaurantFeatures(
            restaurant_id=row['ID'],
            name=row['NAME'],
            url=row.get('RESTAURANT_URL', ''),
            picture_url=row.get('PICTURE', ''),
            star_rating=float(row['RATING']),
            description=row.get('DESCRIPTION', ''),
            latitude=float(row['LATITUDE']),
            longitude=float(row['LONGITUDE']),
            min_price=float(row.get('MIN_PRICE', 0)),
            max_price=float(row.get('MAX_PRICE', 0)),
            price_level=row['PRICE_LEVEL'],
            number_of_tables=int(row.get('TABLES_NUMBER', 0)),
            work_times=eval(row['HOURS']) if 'work_times' in row and isinstance(row['HOURS'], str) else {},
            features=row.get('FEATURES', '').split(',') if pd.notna(row.get('FEATURES')) else [],
            tags=row.get('REVIEW_TAGS', '').split(',') if pd.notna(row.get('REVIEW_TAGS')) else [],
            cuisines=row['CUISINES'].split(','),
            meal_types=row.get('MEAL_TYPES', '').split(',') if pd.notna(row.get('MEAL_TYPES')) else [],
            dishes_with_prices=eval(row['DISHS_WITH_PRICES'])
             if 'dishes_with_prices' in row and isinstance(row['DISHS_WITH_PRICES'], str) else {}
        ))

    # Load interactions
    interactions_df = pd.read_csv(interactions_csv_path)
    interactions = []
    for _, row in interactions_df.iterrows():
        interactions.append({
            'user_id': str(row['CustomerId']),
            'restaurant_id': str(row['RestaurantId']),
            'booking_time': pd.to_datetime(row['DateAndTime'], errors='coerce') if 'booking_time' in row else None,
            'receive_time': pd.to_datetime(row['ReceiveDate'], errors='coerce') if 'receipt_time' in row else None,
            'booked_dishes': eval(row['Dishes']) if 'booked_dishes' in row and isinstance(row['Dishes'], str) else {}
        })

    return users, restaurants, interactions