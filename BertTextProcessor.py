import tensorflow as tf

class BertTextProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def call(self, text_input):
        # For TensorFlow tensors in graph mode
        def tokenize(text):
            text = text.numpy().decode('utf-16') if hasattr(text, 'numpy') else str(text)
            result = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf"
            )
            return result['input_ids'], result['token_type_ids'], result['attention_mask']

        input_ids, token_type_ids, attention_mask = tf.py_function(
            tokenize,
            [text_input],
            [tf.int32, tf.int32, tf.int32]
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