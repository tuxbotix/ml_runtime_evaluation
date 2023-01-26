#%%
import os
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import numpy as np

f1_score_c1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

script_path = os.path.dirname(__file__)
model_path = os.path.join(script_path, "classifier.hdf5")
model_updated_path = os.path.join(script_path, "classifier_updated.hdf5")
model_backbone_path = os.path.join(script_path, "classifier_backbone.hdf5")
multiclass_model_path = os.path.join(script_path, "classifier_multiclass.hdf5")

model = tf.keras.models.load_model(str(model_path), custom_objects={"f1": f1_score_c1})

tf.keras.models.save_model(model, model_updated_path)
_ = tf.keras.models.load_model(str(model_updated_path))

#%%

num_classes = 5  # 4 + 1
f1_score_c_n = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5)

# Freeze the entire model
model.trainable = False
# We want to transfer learn only on this one, so find them and unfreeze them
last_dense_layer_name = "dense_1"
last_dense_layer_idx = -1
last_dense_layer_found = False

# HULKs model has a flatten before the classification dense layers.
# We won't use the flatten layer* -> layers[:,last_backbone_layer_idx]
last_backbone_layer_name = "flatten"
last_backbone_layer_idx = -1

model_layer_count = len(model.layers)

for idx, layer in enumerate(model.layers):
    if layer.name == last_dense_layer_name:
        last_dense_layer_found = True
        last_dense_layer_idx = idx
    if layer.name == last_backbone_layer_name:
        last_backbone_layer_idx = idx
    if last_dense_layer_found:
        layer.trainable = True

dense_layer = model.layers[last_dense_layer_idx]

layer_config = dense_layer.get_config()
layer_config["units"] = num_classes
# new_dense = tf.keras.layers.Dense(
#     name=dense_layer.name,
#     units=num_classes,
#     activation=dense_layer.activation,
#     use_bias=dense_layer.use_bias,
#     # kernel_initializer=dense_layer.kernel_initializer,
#     # bias_initializer=dense_layer.bias_initializer,
#     kernel_constraint=dense_layer.kernel_constraint,
#     bias_constraint=dense_layer.bias_constraint,
# )
new_dense = tf.keras.layers.Dense(**layer_config)

#%%

new_layers = (
    model.layers[:last_dense_layer_idx]
    + [new_dense]
    + model.layers[last_dense_layer_idx + 1 :]
)

backbone = tf.keras.models.Sequential(model.layers[0:last_backbone_layer_idx])

new_model = tf.keras.models.Sequential(new_layers)
#%%
backbone.build()
print("backbone summary")
backbone.summary()
backbone.compile()
tf.keras.models.save_model(backbone, model_backbone_path)
#%%

new_model.build()
print("New Model summary")
new_model.summary()
new_model.compile(metrics=[f1_score_c_n])

# Figure out the data stuff...
# new_model.fit()

tf.keras.models.save_model(new_model, multiclass_model_path)

#%%
with Image.open(
    os.path.join(
        script_path,
        "../../data/ball_sample.png",
    )
) as im:
    a = np.asarray(im)[:, :, 0]
    print(a.shape)
    out = model(np.expand_dims(a, 0))
    new_out = new_model(np.expand_dims(a, 0))

    print(f"out, {out}")
    print(f"new_out, {new_out}")

# %%
