"""
This expands the backbone extracted in patch_classifier.py
"""

#%%
import os
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

f1_score_c1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

script_path = os.path.dirname(__file__)
model_backbone_path = os.path.join(script_path, "classifier_backbone.hdf5")
object_detector_path = os.path.join(script_path, "classifier_multiclass.hdf5")
image_dir = "/home/darshana/HULKs/datasets/SPLObjDetectDatasetV2/trainval/images/"
orig_encoder = tf.keras.models.load_model(str(model_backbone_path))

orig_encoder_input = orig_encoder.layers[0].input[0]
orig_encoder_input_shape = orig_encoder_input.shape
base_tile_size = orig_encoder_input_shape[0]

# Rows, cols, channels (= height, width,channels)
proposed_new_input_shape = [120, 180, 1]

quantize_input_shape = list(
    map(
        lambda x: int(x / base_tile_size) * base_tile_size, proposed_new_input_shape[:2]
    )
)
# rows, cols, channels
new_input_shape = (
    quantize_input_shape[0],
    quantize_input_shape[1],
    proposed_new_input_shape[2],
)

new_batch_input_shape = (
    None,
    new_input_shape[0],
    new_input_shape[1],
    new_input_shape[2],
)

#%%
# Change input shape of first layer and build the model.
first_layer = orig_encoder.layers[0]
config = first_layer.get_config()

if "batch_input_shape" in config:
    config["batch_input_shape"] = new_batch_input_shape
if "input_shape" in config:
    config["input_shape"] = new_input_shape
new_layers = [first_layer.from_config(config)]
for layer in orig_encoder.layers[1:]:  # type: tf.keras.layers.Layer
    new_layer = layer.from_config(layer.get_config())
    new_layers.append(new_layer)

new_encoder = tf.keras.models.Sequential(new_layers)
new_encoder.build()
# Once the model is built, the weights are assigned.
layer: tf.keras.layers.Layer
for idx, layer in enumerate(orig_encoder.layers):  # type: (int, tf.keras.layers.Layer)
    new_encoder.layers[idx].set_weights(layer.get_weights())

new_encoder.summary()
#%%
encoder_last_layer_shape = new_encoder.layers[-1].output[0].shape
print("encoder last layer shape ", encoder_last_layer_shape)
new_decoder = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2DTranspose(
            encoder_last_layer_shape[-1],
            input_shape=encoder_last_layer_shape,
            kernel_size=5,
            strides=2,
            padding="same",
            activation="selu",
        ),
        tf.keras.layers.Conv2DTranspose(
            encoder_last_layer_shape[-1] / 2,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="elu",
        ),
        tf.keras.layers.Conv2DTranspose(
            encoder_last_layer_shape[-1] / 2,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="elu",
        ),
        tf.keras.layers.Conv2DTranspose(
            encoder_last_layer_shape[-1] / 4,
            # 1,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="elu",
        ),
        tf.keras.layers.Conv2D(
            1,
            kernel_size=3,
            # strides=2,
            padding="same",
            activation="sigmoid",
            name="decoder_conv_final",
        ),
    ]
)
new_encoder.trainable = True
new_decoder.trainable = True

input_tensor = new_encoder.layers[0].input
out = input_tensor
for layer in new_encoder.layers:
    out = layer(out)
for layer in new_decoder.layers:
    out = layer(out)

new_autoencoder = tf.keras.models.Model(input_tensor, out)
new_autoencoder.build(input_shape=new_input_shape)
new_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

print("Autoenc summary")
new_autoencoder.summary()

tf.keras.models.save_model(new_autoencoder, "autoencoder.hdf5")
# %%
# Dataset ..
batch_size = 32
image_as_mat_size = (min(new_input_shape[0:2]), max(new_input_shape[0:2]))


def add_rd_targets(image):
    # Training is unsupervised, so labels aren't necessary here. However, we
    # need to add "dummy" targets for rate and distortion.
    # regularized = image[0,:,:,0:1] / 255.0
    regularized = image[0, :, :, 0:1]
    return regularized, regularized


dataset_images_only: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    image_dir,
    labels=None,
    validation_split=0.2,
    subset="training",
    seed=123,
    # height, width order
    image_size=image_as_mat_size,
    batch_size=batch_size,
    shuffle=True,
)
dataset_images_only = (
    dataset_images_only.map(add_rd_targets).batch(batch_size).prefetch(8),
)
#%%
# Train

new_autoencoder.fit(
    dataset_images_only,
    epochs=5,
    verbose=1,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="/tmp/autoencoder")],
)


#%%
val = dataset_images_only[0].as_numpy_iterator().next()[0]
print(val.shape)
# with Image.open(
#     os.path.join(
#         script_path,
#         "../../data/01024_GermanOpen2019_HULKs_Sabretooth-2nd_52240197_upper-002.png",
#     )
# ) as im:

# width, height order
# resized = im.resize((image_as_mat_size[1], image_as_mat_size[0]))
# a = np.expand_dims(np.asarray(resized,dtype=float)[:, :, 0:1],0) / 255.0
# print(a.shape, np.max(a.flatten()))
# out = new_autoencoder(a)
enc_out = new_encoder(val)

plt.figure()
plt.imshow(val[0])

out = np.zeros((30, 40))
feature_count = enc_out.shape[-1]

for feature_id in range(
    0,
):
    d = np.linalg.norm(enc_out[0, :, :, feature_id : feature_id + 1], axis=2)
    # plt.figure()
    # plt.imshow(d)
    out = out + d

out /= feature_count

plt.figure()
plt.imshow(d)

out = new_autoencoder(val)
plt.figure()
plt.imshow(out[0])

# %%
