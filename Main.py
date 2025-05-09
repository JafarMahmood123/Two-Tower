import pickle
import tensorflow as tf
from DataLoader import load_data_from_csv
from DataProcessing import create_features, feature_generator, create_restaurant_dataset
from TwoTowerModel import TwoTowerModel

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

    # Create datasets
    restaurant_dataset = create_restaurant_dataset(restaurants)
    restaurant_dataset = restaurant_dataset.map(
        lambda x: {
            'restaurant_id': x['restaurant_id'],
            'star_rating': tf.cast(x['star_rating'], tf.float32),
            'min_price': tf.cast(x['min_price'], tf.float32),
            'max_price': tf.cast(x['max_price'], tf.float32),
            'latitude': tf.cast(x['latitude'], tf.float32),
            'longitude': tf.cast(x['longitude'], tf.float32),
            'description': tf.ensure_shape(x['description'], ())
        }
    )

    # Create interaction features
    features_list = []
    failed_ids = set()
    for interaction in interactions:
        feat = create_features(interaction, restaurants)
        if feat is None:
            failed_ids.add(str(interaction['restaurant_id']))
        else:
            features_list.append(feat)

    if not features_list:
        raise ValueError("No valid features generated")

    # Create a generator function that closes over features_list
    def wrapped_feature_generator():
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
        wrapped_feature_generator,
        output_signature={
            'user_id': tf.TensorSpec(shape=(), dtype=tf.string),
            'star_rating': tf.TensorSpec(shape=(), dtype=tf.float32),
            'min_price': tf.TensorSpec(shape=(), dtype=tf.float32),
            'max_price': tf.TensorSpec(shape=(), dtype=tf.float32),
            'latitude': tf.TensorSpec(shape=(), dtype=tf.float32),
            'longitude': tf.TensorSpec(shape=(), dtype=tf.float32),
            'description': tf.TensorSpec(shape=(), dtype=tf.string)
        }
    ).batch(32).prefetch(tf.data.AUTOTUNE)

    # Train/test split
    train_size = int(0.8 * len(features_list))
    train_data = features_dataset.take(train_size)
    test_data = features_dataset.skip(train_size)

    # Initialize and compile model
    model = TwoTowerModel(users, restaurant_dataset)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    # Train
    print("\n=== Training ===")
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.TensorBoard()
        ]
    )

    # Evaluate
    print("\n=== Evaluation ===")
    results = model.evaluate(test_data, return_dict=True)
    print("Evaluation Results:", results)

    # Save
    print("\n=== Saving Model ===")
    model.save('restaurant_recommender_bert')
    with open('model_assets.pkl', 'wb') as f:
        pickle.dump({
            'user_id_to_idx': {u.user_id: i for i, u in enumerate(users)},
            'restaurant_id_to_idx': {r.restaurant_id: i for i, r in enumerate(restaurants)},
            'restaurant_data': [r.to_dict() for r in restaurants]
        }, f)

    print("\n=== Training Complete ===")