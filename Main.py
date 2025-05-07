import tensorflow as tf
import tensorflow.keras
import tensorflow_recommenders as tfrs
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from tensorflow.keras import layers
import pandas as pd
import pickle
from transformers import BertTokenizer, TFBertModel

# 1. Data Structure Definitions
class UserFeatures:
    def __init__(self, user_id: str, first_name: str, last_name: str,
                 birthdate: str, email: str, password: str,
                 Country: str, City: str):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.birthdate = datetime.strptime(birthdate, '%Y-%m-%d')
        self.age = (datetime.now() - self.birthdate).days // 365
        self.email = email
        self.password = password
        self.country = Country  # New attribute
        self.city = City  # New attribute

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'age': self.age,
            'first_name': self.first_name,
            'last_name': self.last_name
        }


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


# 2. BERT Text Processor Layer
class BertTextProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def call(self, text_input):
        # Process text through tokenizer
        def tokenize(text):
            result = self.tokenizer(
                text.numpy().decode('utf-8'),
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf"
            )
            return result['input_ids'], result['token_type_ids'], result['attention_mask']

        # Convert to tensors
        input_ids, token_type_ids, attention_mask = tf.py_function(
            tokenize,
            [text_input],
            Tout=[tf.int32, tf.int32, tf.int32]
        )

        # Set shapes explicitly
        input_ids.set_shape([None, self.max_length])
        token_type_ids.set_shape([None, self.max_length])
        attention_mask.set_shape([None, self.max_length])

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }

    def compute_output_signature(self, input_signature):
        return {
            'input_ids': tf.TensorSpec(shape=(None, self.max_length), dtype=tf.int32),
            'token_type_ids': tf.TensorSpec(shape=(None, self.max_length), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None, self.max_length), dtype=tf.int32)
        }


class TwoTowerModel(tfrs.models.Model):
    def __init__(self, users, restaurant_dataset):
        super().__init__()

        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.bert_text_processor = BertTextProcessor(self.tokenizer)

        # User Tower
        self.user_model = tf.keras.Sequential([
            layers.Embedding(
                input_dim=len(users) + 1,
                output_dim=64,
                name="user_embedding"
            ),
            layers.Dense(64, activation='relu'),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])

        # Restaurant Tower - Numeric Branch
        self.numeric_branch = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')
        ])

        # Restaurant Tower - Text Branch
        self.text_branch = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')
        ])

        # Combined Tower
        self.combined = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])

        # Preprocess candidate embeddings
        self.candidates = restaurant_dataset.map(self.restaurant_model).cache()

        # Task setup
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.candidates
            ),
            negative_sampling_ratio=4
        )

    def restaurant_model(self, features):
        # Process numeric features
        numeric_features = tf.stack([
            tf.expand_dims(features['star_rating'], -1),
            tf.expand_dims(features['min_price'], -1),
            tf.expand_dims(features['max_price'], -1),
            tf.expand_dims(features['latitude'], -1),
            tf.expand_dims(features['longitude'], -1)
        ], axis=1)

        numeric_embedding = self.numeric_branch(numeric_features)

        # Process text features
        if isinstance(features['description'], str):
            text_input = features['description']
        else:
            text_input = tf.strings.as_string(features['description'])

        tokenized = self.bert_text_processor(text_input)
        bert_output = self.bert_model(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            token_type_ids=tokenized['token_type_ids']
        )
        text_embedding = self.text_branch(bert_output.last_hidden_state[:, 0, :])

        return self.combined(tf.concat([numeric_embedding, text_embedding], axis=1))


# 4. Data Loading and Preparation
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


if __name__ == "__main__":
    # Load data
    users, restaurants, interactions = load_data_from_csv(
        'customer_profiles_with_roles.csv',
        'cleaned_restaurant_data.csv',
        'generated_restaurant_orders.csv'
    )

    # Data statistics
    print(f"\n=== Data Statistics ===")
    print(f"Number of users: {len(users)}")
    print(f"Number of restaurants: {len(restaurants)}")
    print(f"Number of interactions: {len(interactions)}")

    # Sample records
    print("\n=== Sample Records ===")
    print("User:", users[0].to_dict())
    print("Restaurant:", restaurants[0].to_dict())
    print("Interaction:", interactions[0])

    # Feature processing function
    def create_features(interaction):
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

# Usage:
restaurant_dataset = create_restaurant_dataset(restaurants)


# Create and verify dataset
print("\n=== Creating Dataset ===")
failed_ids = set()  # Track failed restaurant IDs
features_list = []

for interaction in interactions:
    feat = create_features(interaction)
    if feat is None:
        restaurant_id = str(interaction['restaurant_id'])
        failed_ids.add(restaurant_id)
    else:
        features_list.append(feat)

if not features_list:
    print("\n=== ERROR DETAILS ===")
    print("No valid features generated - check your data processing")
    print(f"Total interactions processed: {len(interactions)}")
    print(f"Failed interactions: {len(failed_ids)}")
    print("Restaurant IDs that weren't found:")
    for i, rid in enumerate(sorted(failed_ids), 1):
        print(f"{i}. {rid}")
        if i >= 20:  # Limit to first 20 IDs
            print(f"... and {len(failed_ids)-20} more")
            break
    raise ValueError("No valid features generated - see above for missing restaurant IDs")

def feature_generator():
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

features_dataset = tf.data.Dataset.from_generator(
    feature_generator,
    output_signature={
        'user_id': tf.TensorSpec(shape=(), dtype=tf.string),
        'star_rating': tf.TensorSpec(shape=(), dtype=tf.float32),
        'min_price': tf.TensorSpec(shape=(), dtype=tf.float32),
        'max_price': tf.TensorSpec(shape=(), dtype=tf.float32),
        'latitude': tf.TensorSpec(shape=(), dtype=tf.float32),
        'longitude': tf.TensorSpec(shape=(), dtype=tf.float32),
        'description': tf.TensorSpec(shape=(), dtype=tf.string)
    }
)
features_dataset = features_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# # Dataset verification
# print("\n=== Dataset Verification ===")
# print(f"Total batches: {len(list(features_dataset))}")
# sample_batch = next(iter(features_dataset))
# print("\nSample Batch:")
# for key, value in sample_batch.items():
#     print(f"{key}: {value.shape} ({value.dtype})")

# Train/test split
train_size = int(0.8 * len(features_list))
train_data = features_dataset.take(train_size)
test_data = features_dataset.skip(train_size)

# print(f"\nTrain batches: {len(list(train_data))}")
# print(f"Test batches: {len(list(test_data))}")


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


# Convert restaurants to a format suitable for the candidate set
# restaurant_features = [r.to_dict() for r in restaurants]
# restaurant_dataset = tf.data.Dataset.from_tensor_slices(restaurant_features)

restaurant_dataset = restaurant_dataset.map(
    lambda x: {
        'restaurant_id': tf.strings.as_string(x['restaurant_id']),
        'star_rating': tf.cast(x['star_rating'], tf.float32),
        'min_price': tf.cast(x['min_price'], tf.float32),
        'max_price': tf.cast(x['max_price'], tf.float32),
        'latitude': tf.cast(x['latitude'], tf.float32),
        'longitude': tf.cast(x['longitude'], tf.float32),
        'description': tf.strings.as_string(x['description'])
    }
)

# Print a sample batch from the dataset
sample_batch = next(iter(restaurant_dataset))
print("Sample batch:", sample_batch)
for key, value in sample_batch.items():
    print(f"{key}: {value} (type: {type(value)})")


# Initialize model
print("\n=== Model Initialization ===")
model = TwoTowerModel(users, restaurant_dataset)  # Use the cleaned dataset
model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
print(model.summary())

# Training
print("\n=== Training ===")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    verbose=2,  # Explicit progress bar
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.TensorBoard()
    ]
)

# Evaluation
print("\n=== Evaluation ===")
results = model.evaluate(test_data, return_dict=True)
print("Evaluation Results:", results)

# Saving
print("\n=== Saving Model ===")
model.save('restaurant_recommender_bert')
with open('model_assets.pkl', 'wb') as f:
    pickle.dump({
        'user_id_to_idx': {u.user_id: i for i, u in enumerate(users)},
        'restaurant_id_to_idx': {r.restaurant_id: i for i, r in enumerate(restaurants)},
        'restaurant_data': [r.to_dict() for r in restaurants]
    }, f)

print("\n=== Training Complete ===")


