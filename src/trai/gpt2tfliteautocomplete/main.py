"""
https://colab.research.google.com/github/tensorflow/codelabs/blob/main/KerasNLP/io2023_workshop.ipynb#scrollTo=pGVjcn4tOxKP

pip install -q git+https://github.com/keras-team/keras-nlp.git@google-io-2023 tensorflow-text==2.12
"""

import numpy as np
import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from tensorflow import keras
from tensorflow.lite.python import interpreter
import time
from nltk import tokenize
import nltk
import progressbar


# GET Initial MODEL (TF.KERAS) ----------------------------------------------------------------------------------------
def get_gpt2_tokenizer():
    gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")
    return gpt2_tokenizer


def get_gpt2_preprocessor():
    gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=256,
        add_end_token=True,
    )
    return gpt2_preprocessor


def get_gpt2__base_model():
    gpt2_preprocessor = get_gpt2_preprocessor()
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en", preprocessor=gpt2_preprocessor)
    return gpt2_lm


def get_gpt2_cnn_model():
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en_cnn_dailymail")
    return gpt2_lm


def save_tf_model(model, name: str):
    model.backbone.save_weights(f"{name}.h5")


def run_tf_model(tf_model, text: str, max_length: int = 200):
    start = time.time()
    output = tf_model.generate(text, max_length=max_length)
    print("\nModel output:")
    print(output.numpy().decode("utf-8"))
    end = time.time()
    print("TOTAL TIME ELAPSED: ", end - start)


# TRAIN on CNN DATASET ------------------------------------------------------------------------------------------------
def download_cnn_dataset():
    start = time.time()
    cnn_ds = tfds.load('cnn_dailymail', as_supervised=True)
    end = time.time()
    print("TOTAL TIME ELAPSED: ", end - start)
    return cnn_ds


def inspect_cnn_dataset(cnn_ds):
    for article, highlights in cnn_ds['train']:
        print(article.numpy())
        print(highlights.numpy())
        break


def merge_sentences(sentences, max_length):
    res = []
    cur_len = 0
    cur_sentences = []
    for s in sentences:
        if cur_len + len(s) > max_length:
            # If adding the next sentence exceeds `max_length`, we add the
            # current sentences into collection
            res.append(" ".join(cur_sentences))
            cur_len = len(s)
            cur_sentences = [s]
        else:
            cur_len += len(s)
            cur_sentences.append(s)
    res.append(" ".join(cur_sentences))
    return res


def get_training_set(cnn_ds):
    max_length = 512
    all_sentences = []
    count = 0
    total = len(cnn_ds["train"])
    num_articles_to_process = 20000
    progressbar_update_freq = 2000

    widgets = [' [',
               progressbar.Timer(format='elapsed time: %(elapsed)s'),
               '] ',
               progressbar.Bar('*'), ' (',
               progressbar.ETA(), ') ',
               ]

    # Render a progressbar to track progress
    bar = progressbar.ProgressBar(
        max_value=num_articles_to_process // progressbar_update_freq + 2,
        widgets=widgets).start()

    for article, highlight in cnn_ds['train']:
        # Use NLTK tokenize to split articles into sentences
        sentences = tokenize.sent_tokenize(str(article))
        # Merge individual sentences into longer context
        combined_res = merge_sentences(sentences, max_length)
        # Add merged context into collection
        all_sentences.extend(combined_res)
        count += 1
        if count % progressbar_update_freq == 0:
            bar.update(count / progressbar_update_freq)
        if count >= num_articles_to_process:
            break

    gpt2_preprocessor = get_gpt2_preprocessor()
    tf_train_ds = tf.data.Dataset.from_tensor_slices(all_sentences)
    processed_ds = tf_train_ds.map(gpt2_preprocessor, tf.data.AUTOTUNE).batch(20).cache().prefetch(tf.data.AUTOTUNE)
    part_of_ds = processed_ds.take(100)
    return part_of_ds


def train_tf_model(model, train_set):
    model.include_preprocessing = False

    num_epochs = 1

    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps=train_set.cardinality() * num_epochs,
        end_learning_rate=0.0,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=keras.optimizers.experimental.Adam(lr),
        loss=loss,
        weighted_metrics=["accuracy"])

    model.fit(train_set, epochs=num_epochs)
    return model


# TFLite --------------------------------------------------------------------------------------------------------------
def tf_model_to_concrete_function(model):
    @tf.function
    def generate(prompt, max_length):
        return model.generate(prompt, max_length)

    concrete_func = generate.get_concrete_function(tf.TensorSpec([], tf.string), 100)
    return concrete_func


def model_to_tflite(tf_model, concrete_func, quant: bool = False):
    tf_model.jit_compile = False
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], tf_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.allow_custom_ops = True
    if quant:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.experimental_select_user_tf_ops = ["UnsortedSegmentJoin", "UpperBound"]
    converter._experimental_guarantee_all_funcs_one_use = True
    generate_tflite = converter.convert()
    return generate_tflite


def save_tfl_model(generate_tflite, name:str):
    with open(f'{name}.tflite', 'wb') as f:
        f.write(generate_tflite)


def load_tfl_model(name: str):
    with open(f'{name}.tflite', 'rb') as f:
        tfl_model = f.read()
    return tfl_model


def run_tfl_model(input_text: str, generate_tflite):
    interp = interpreter.InterpreterWithCustomOps(
        model_content=generate_tflite,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    interp.get_signature_list()

    generator = interp.get_signature_runner('serving_default')
    output = generator(prompt=np.array([input_text]))
    print("\nGenerated with TFLite:\n", output["output_0"])


# TEST ================================================================================================================
def run_base_model():
    m = get_gpt2__base_model()
    run_tf_model(m, "My trip to Yosemite was")
    run_tf_model(m, "That Italian restaurant is")


def fine_tune_on_cnn_dataset():
    m = get_gpt2__base_model()
    ds = download_cnn_dataset()
    inspect_cnn_dataset(ds)
    nltk.download('punkt')
    ts = get_training_set(ds)
    m = train_tf_model(m, ts)
    run_tf_model(m, "Breaking news: the river")
    save_tf_model(m, 'finetuned_model')


def run_cnn_model():
    m = get_gpt2_cnn_model()
    run_tf_model(m, "Breaking news: the river")


def convert_to_tflite_std():
    m = get_gpt2_cnn_model()
    concrete_function = tf_model_to_concrete_function(m)
    generate_tflite = model_to_tflite(m, concrete_function)
    # run_tfl_model("I'm enjoying a", generate_tflite)
    save_tfl_model(generate_tflite, 'unquantized_gpt2')


def convert_to_tflite_quantized():
    m = get_gpt2_cnn_model()
    concrete_function = tf_model_to_concrete_function(m)
    generate_tflite = model_to_tflite(m, concrete_function, quant=True)
    # run_tfl_model("I'm enjoying a", generate_tflite)
    save_tfl_model(generate_tflite, 'quantized_gpt2')


def run_tflite_std():
    m = load_tfl_model('unquantized_gpt2')
    run_tfl_model("I'm enjoying a", m)


def run_tflite_quant():
    m = load_tfl_model('quantized_gpt2')
    run_tfl_model("I'm enjoying a", m)


if __name__ == '__main__':
    fine_tune_on_cnn_dataset()
