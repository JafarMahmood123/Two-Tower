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
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased',
                                                      trainable=True)  # Ensure BERT is trainable

        # User Tower
        self.user_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=[u.user_id for u in users],
            mask_token=None,
            num_oov_indices=1
        )

        self.user_model = tf.keras.Sequential([
            self.user_id_lookup,
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
            layers.Dense(64, activation='relu')
        ])

        # Restaurant Tower - Text Branch
        self.text_branch = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu')
        ])

        # Combined Tower - MUST MATCH USER TOWER DIMENSION
        self.combined = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),  # Changed from 128 to 64
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])

        # Properly cache candidates
        self.candidates = restaurant_dataset.batch(128).map(self.restaurant_model).cache()

        # Task setup
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.candidates
            )
        )

    def restaurant_model(self, features):
        # Process numeric features with batch dimension
        numeric_features = tf.stack([
            tf.expand_dims(features['star_rating'], -1),
            tf.expand_dims(features['min_price'], -1),
            tf.expand_dims(features['max_price'], -1),
            tf.expand_dims(features['latitude'], -1),
            tf.expand_dims(features['longitude'], -1)
        ], axis=1)

        numeric_embedding = self.numeric_branch(numeric_features)

        # Process text features with proper batch handling
        def process_text(text):
            if isinstance(text, bytes):
                return text.decode('utf-8')
            return str(text)

        def tokenize(text):
            text_str = tf.py_function(process_text, [text], tf.string)
            tokenized = self.tokenizer(
                text_str.numpy(),
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors="tf"
            )
            return tokenized['input_ids'], tokenized['token_type_ids'], tokenized['attention_mask']

        input_ids, token_type_ids, attention_mask = tf.py_function(
            tokenize,
            [features['description']],
            [tf.int32, tf.int32, tf.int32]
        )

        # Ensure proper shapes
        batch_size = tf.shape(numeric_embedding)[0]
        input_ids.set_shape([None, 128])
        token_type_ids.set_shape([None, 128])
        attention_mask.set_shape([None, 128])

        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Ensure text embedding matches batch size
        text_embedding = self.text_branch(bert_output.last_hidden_state[:, 0, :])
        text_embedding = tf.ensure_shape(text_embedding, [None, 64])

        # Safe concatenation
        combined_features = tf.concat([
            tf.reshape(numeric_embedding, [-1, 64]),
            tf.reshape(text_embedding, [-1, 64])
        ], axis=1)

        return self.combined(combined_features)

    def call(self, inputs):
        """Forward pass with proper batch handling"""
        user_embeddings = self.user_model(inputs["user_id"])
        restaurant_embeddings = self.restaurant_model(inputs)
        return user_embeddings, restaurant_embeddings

    def compute_loss(self, features, training=False):
        """Compute loss with dimension checks"""
        user_embeddings = self.user_model(features["user_id"])
        positive_embeddings = self.restaurant_model(features)
        return self.task(user_embeddings, positive_embeddings)