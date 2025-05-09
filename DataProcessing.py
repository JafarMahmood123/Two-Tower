import tensorflow as tf


def create_features(interaction, restaurants):
    try:
        # print(f"\nProcessing interaction: {interaction['user_id']} -> {interaction['restaurant_id']}")  # DEBUG

        # Convert both IDs to string for consistent comparison
        restaurant_id = str(interaction['restaurant_id'])
        restaurant = next(
            r for r in restaurants
            if str(r.restaurant_id) == restaurant_id
        )

        # print(f"Found matching restaurant: {restaurant.name}")  # DEBUG
        return {
            'user_id': interaction['user_id'],
            'restaurant_id': interaction['restaurant_id'],
            'star_rating': restaurant.star_rating,
            'min_price': restaurant.min_price,
            'max_price': restaurant.max_price,
            'latitude': restaurant.latitude,
            'longitude': restaurant.longitude,
            'description': (
                f"{restaurant.name}. {restaurant.description}. "
                f"Cuisines: {', '.join(restaurant.cuisines)}"
            )
        }
    except StopIteration:
        print(f"ERROR: No restaurant found for ID {restaurant_id}")
        return None
    except Exception as e:
        print(f"ERROR processing interaction: {str(e)}")
        return None



def create_restaurant_dataset(restaurants):
    """Convert restaurant data to a TensorFlow-compatible format"""
    # Extract only the numeric/text features needed for the model
    features = {
        'restaurant_id': [],
        'star_rating': [],
        'min_price': [],
        'max_price': [],
        'latitude': [],
        'longitude': [],
        'description': []
    }

    for r in restaurants:
        # Create the description string
        desc = f"{r.name}. {r.description}. Cuisines: {', '.join(r.cuisines)}"

        # Add to features dictionary
        features['restaurant_id'].append(str(r.restaurant_id))
        features['star_rating'].append(float(r.star_rating))
        features['min_price'].append(float(r.min_price))
        features['max_price'].append(float(r.max_price))
        features['latitude'].append(float(r.latitude))
        features['longitude'].append(float(r.longitude))
        features['description'].append(desc)

    # Convert to TensorFlow Dataset
    return tf.data.Dataset.from_tensor_slices({
        'restaurant_id': tf.constant(features['restaurant_id']),
        'star_rating': tf.constant(features['star_rating'], dtype=tf.float32),
        'min_price': tf.constant(features['min_price'], dtype=tf.float32),
        'max_price': tf.constant(features['max_price'], dtype=tf.float32),
        'latitude': tf.constant(features['latitude'], dtype=tf.float32),
        'longitude': tf.constant(features['longitude'], dtype=tf.float32),
        'description': tf.constant(features['description'])
    })



def feature_generator(features_list):
    for feature in features_list:
        yield {
            'user_id': feature['user_id'],
            'star_rating': feature['star_rating'],
            'min_price': feature['min_price'],
            'max_price': feature['max_price'],
            'latitude': feature['latitude'],
            'longitude': feature['longitude'],
            'description': feature['description']
        }



def restaurant_generator(restaurants):
    for r in restaurants:
        try:
            r_dict = r.to_dict()
            desc = (
                f"{r_dict.get('name', '')}. {r_dict['description']}. "
                f"Cuisines: {', '.join(r_dict.get('cuisines', []))}"
            )
            # Ensure numeric fields are valid
            yield {
                'restaurant_id': str(r_dict['restaurant_id']),
                'star_rating': float(r_dict['star_rating']),
                'min_price': float(r_dict['min_price']),
                'max_price': float(r_dict['max_price']),
                'latitude': float(r_dict['latitude']),
                'longitude': float(r_dict['longitude']),
                'description': str(desc)  # Explicitly convert to string
            }
        except Exception as e:
            print(f"Skipping invalid restaurant {r.restaurant_id}: {e}")
            continue