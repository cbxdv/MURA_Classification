import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.utils import load_img, img_to_array

input_shape = (224, 224, 3)

dense201 = DenseNet201(include_top=False, input_shape=input_shape, weights='imagenet', pooling="max", classes=2)
inputs = tf.keras.layers.Input(shape=input_shape, name='in1')
x = dense201(inputs)
x = tf.keras.layers.Dropout(rate=0.2, name='d1')(x)
output = tf.keras.layers.Dense(units=1, activation='sigmoid', name='o1')(x)

model = tf.keras.Model(inputs, output)
model.load_weights('dense201_model2.keras')


def classify_image(inp: str):
    img = load_img(inp, target_size=(224, 224, 3))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    pred = model(x)
    prob = round(float(pred.numpy()[0][0]), 3)
    if prob >= 0.5:
        return f"Abnormality Detected ⚠️ ({prob})"
    return f"Normal ✅ ({prob})"


gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='filepath', label='Radiograph'),
    outputs=gr.Label(label='Prediction'),
    examples=[
        "MURA-v1.1//train//XR_HAND//patient09811//study1_positive//image2.png",
        "MURA-v1.1/train/XR_FOREARM/patient09558/study1_negative/image1.png",
        'MURA-v1.1/train/XR_WRIST/patient08618/study1_negative/image3.png',
        'MURA-v1.1/train/XR_WRIST/patient07427/study1_positive/image1.png',
        'MURA-v1.1/train/XR_FOREARM/patient09258/study1_positive/image2.png',
        'MURA-v1.1/train/XR_HAND/patient11063/study1_negative/image3.png'
    ],
    allow_flagging='never'
).launch()