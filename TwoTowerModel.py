import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import logging

# Reduce BERT warning verbosity
logging.getLogger("transformers").setLevel(logging.ERROR)


class TwoTowerModel(tfrs.models.Model):
    def __init__(self, users, restaurant_dataset):
        super().__init__()

        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased', trainable=True)

        # User Tower
        self.user_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=[u.user_id for u in users],
            mask_token=None,
            num_oov_indices=1
        )

        self.user_model = tf.keras.Sequential([
            self.user_id_lookup,
            layers.Embedding(input_dim=len(users) + 1, output_dim=64, name="user_embedding"),
            layers.Dense(64, activation='relu'),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])

        # Restaurant Tower Components
        # Numeric Branch
        # In __init__ method:
        self.numeric_branch = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu')  # Remove the pooling layer
        ])

        # Text Branch
        self.text_branch = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu')
        ])

        # Combined Tower
        self.combined = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])

        # Precompute candidate embeddings
        self._preprocess_candidates(restaurant_dataset)

        # Initialize retrieval task
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.candidate_embeddings
            )
        )

    def _preprocess_candidates(self, restaurant_dataset):
        """Preprocess restaurant data and cache embeddings"""
        # Process and cache candidate dataset
        self.candidates = restaurant_dataset.batch(128).cache()

        # Dummy input to initialize layers
        dummy_input = {
            'restaurant_id': tf.constant(["dummy_id"]),
            'star_rating': tf.constant([0.0], dtype=tf.float32),
            'min_price': tf.constant([0.0], dtype=tf.float32),
            'max_price': tf.constant([0.0], dtype=tf.float32),
            'latitude': tf.constant([0.0], dtype=tf.float32),
            'longitude': tf.constant([0.0], dtype=tf.float32),
            'description': tf.constant(["dummy description"])
        }
        _ = self.restaurant_model(dummy_input)  # Force layer initialization

        # Precompute all candidate embeddings
        self.candidate_embeddings = self.candidates.map(self.restaurant_model).cache()

    def restaurant_model(self, features):
        # In restaurant_model method:
        # Replace the numeric_features stacking with:
        numeric_features = tf.concat([
            tf.expand_dims(features['star_rating'], -1),
            tf.expand_dims(features['min_price'], -1),
            tf.expand_dims(features['max_price'], -1),
            tf.expand_dims(features['latitude'], -1),
            tf.expand_dims(features['longitude'], -1)
        ], axis=1)

        numeric_embedding = self.numeric_branch(numeric_features)

        # Process text features
        def _bert_tokenize(text):
            # Convert batch of byte strings to regular strings
            texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t)
                     for t in text.numpy()]

            # Batch tokenization
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors="tf"
            )
            return (
                tokenized['input_ids'],
                tokenized['token_type_ids'],
                tokenized['attention_mask']
            )

        # Convert to tensors with proper shapes
        input_ids, token_type_ids, attention_mask = tf.py_function(
            _bert_tokenize,
            [features['description']],
            [tf.int32, tf.int32, tf.int32]
        )
        input_ids.set_shape([None, 128])
        token_type_ids.set_shape([None, 128])
        attention_mask.set_shape([None, 128])

        # BERT processing
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_embedding = self.text_branch(bert_output.last_hidden_state[:, 0, :])

        # Combine features
        combined_features = tf.concat([
            numeric_embedding,
            text_embedding
        ], axis=1)

        return self.combined(combined_features)

    def call(self, inputs):
        user_embeddings = self.user_model(inputs["user_id"])
        restaurant_embeddings = self.restaurant_model(inputs)
        return user_embeddings, restaurant_embeddings

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        positive_embeddings = self.restaurant_model(features)
        return self.task(user_embeddings, positive_embeddings)