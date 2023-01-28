"""
This expands the backbone extracted in patch_classifier.py
"""

#%%
import argparse
import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple


def configure_gpu(memory_limit: int):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)],
                )
            tf.config.set_visible_devices([], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


script_path = os.path.dirname(__file__)
print("Imported libraries")

#%%
def load_networks(backbone_path: os.PathLike):
    return tf.keras.models.load_model(str(backbone_path))


#%%
def reshape_backbone_inputs(
    proposed_new_input_shape: list[int],
) -> Tuple[Tuple[int], Tuple[int]]:

    print("setting up new shape")
    # base_tile_size = original_input_shape[0]
    base_tile_size = 32

    quantize_input_shape = list(
        map(
            lambda x: int(x / base_tile_size) * base_tile_size,
            proposed_new_input_shape[:2],
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
    return new_input_shape, new_batch_input_shape


#%%
def get_backbone_with_new_shape(
    new_input_shape: Tuple[int],
    new_batch_input_shape: Tuple[int],
    original_encoder: tf.keras.Model,
) -> tf.keras.Model:
    # Change input shape of first layer and build the model.
    print("Make a new encoder with the new input shape")
    first_layer = original_encoder.layers[0]
    config = first_layer.get_config()

    if "batch_input_shape" in config:
        config["batch_input_shape"] = new_batch_input_shape
    if "input_shape" in config:
        config["input_shape"] = new_input_shape
    new_layers = [first_layer.from_config(config)]
    for layer in original_encoder.layers[1:]:  # type: tf.keras.layers.Layer
        new_layer = layer.from_config(layer.get_config())
        new_layers.append(new_layer)

    new_encoder = tf.keras.models.Sequential(new_layers, name="encoder")
    new_encoder.build()
    # Once the model is built, the weights are assigned.
    layer: tf.keras.layers.Layer
    for idx, layer in enumerate(
        original_encoder.layers
    ):  # type: (int, tf.keras.layers.Layer)
        new_encoder.layers[idx].set_weights(layer.get_weights())

    new_encoder.summary()
    return new_encoder


#%%
def get_simple_decoder(
    new_encoder: tf.keras.models.Model, dropout_rate: float
) -> tf.keras.Model:
    """
    This is a naive decoder made by observing the encoder..
    """
    encoder_first_layer_input_shape = new_encoder.layers[0].input.shape
    decoder_input_tensor = new_encoder.layers[-1].output
    decoder_input_tensor_depth = decoder_input_tensor.shape[-1]
    # def decoder_block(encoder_output):
    #         x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(    encoder_last_layer_output)
    #         x = tf.keras.layers.UpSampling2D((2, 2))(x)

    # Block 1
    x = tf.keras.layers.Conv2D(
        decoder_input_tensor_depth, kernel_size=5, activation="selu", padding="same"
    )(decoder_input_tensor)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    # Block 2
    x = tf.keras.layers.Conv2D(
        decoder_input_tensor_depth / 2, (3, 3), activation="elu", padding="same"
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    # Block 3
    x = tf.keras.layers.Conv2D(
        decoder_input_tensor_depth / 2, (3, 3), activation="elu", padding="same"
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    # Block 4
    x = tf.keras.layers.Conv2D(
        decoder_input_tensor_depth / 4, (3, 3), activation="elu", padding="same"
    )(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    # Output
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    print(f"AutoEnc shapes {decoded.shape} ==? {encoder_first_layer_input_shape}")
    assert decoded.shape == encoder_first_layer_input_shape

    return tf.keras.models.Model(decoder_input_tensor, decoded, name="decoder")


def get_new_autoencoder(
    encoder_trainable: bool,
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    compile_parameters: dict = {
        "optimizer": "adam",
        "loss": tf.keras.losses.MeanAbsoluteError(),
    },
) -> tf.keras.Model:
    print("Construct the autoencoder")
    encoder.trainable = encoder_trainable
    decoder.trainable = True

    # Build the autoencoder with the functional API (call each layer as a function with inputs)
    input_tensor = encoder.layers[0].input
    decoder_out = decoder(encoder(input_tensor))

    new_autoencoder = tf.keras.models.Model(
        input_tensor, decoder_out, name="autoencoder"
    )
    new_autoencoder.build(input_shape=input_tensor.shape)
    new_autoencoder.compile(**compile_parameters)

    print("Autoencoder summary")
    new_autoencoder.summary()
    return new_autoencoder


# %%
def setup_dataset(
    image_shape,
    image_directory: os.PathLike,
    batch_size: int,
    seed: int,
    get_yuv: bool = False,
    get_grayscale: bool = True,
    regularize: bool = True,
    augment: bool = False,
    dataset_parameters: dict = {
        "labels": None,
        # "validation_split": 0.0,
        # "subset": "training",
        "shuffle": True,
    },
) -> tf.data.Dataset:
    # Dataset ..
    print("Setting up dataset")
    channels = image_shape[2]
    if get_grayscale and channels != 1:
        print(
            "Requested grayscale, but channel count needed is 1!. Ignoring grayscale conversion"
        )
    get_grayscale = get_grayscale and channels == 1

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed),
        ]
    )

    def add_rd_targets(image):
        """
        Training is unsupervised, so labels aren't necessary here. However, we
        need to add "dummy" targets for rate and distortion.
        """
        if regularize:
            image = image / 255.0
        if get_yuv and not get_grayscale:
            image = tf.image.rgb_to_yuv(image)
        elif get_grayscale:
            image = tf.image.rgb_to_grayscale(image)
        if augment:
            image = data_augmentation(image)
        image = image[0, :, :, 0:channels]
        return image, image

    dataset_parameters["directory"] = image_directory
    dataset_parameters["image_size"] = (image_shape[0], image_shape[1])
    dataset_parameters["batch_size"] = batch_size
    dataset_parameters["seed"] = seed
    dataset_images_only: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        **dataset_parameters
    )
    return (
        dataset_images_only.map(add_rd_targets)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


#%%
# Train
def run_training(
    model: tf.keras.Model,
    dataset_images_only: tf.data.Dataset,
    epochs: int,
    verbose: bool,
    log_dir="/tmp/autoencoder",
):
    print("Train")
    model.fit(
        dataset_images_only,
        epochs=epochs,
        verbose=verbose,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)],
    )


#%%
def show_simple_results(
    encoder: tf.keras.Model,
    autoencoder: tf.keras.Model,
    dataset: tf.data.Dataset,
    batch_count: int,
    output_dir,
    image_prefix="fig",
):
    print("get some outputs.")
    for idx in range(0, batch_count):
        val = dataset.as_numpy_iterator().next()[0]
        # Original encoder is not trained with normalized [0,1] range!
        enc_out = encoder(val)
        # last axis has the 32 features per pixel, first axis is the batch.
        enc_out_summed = np.sum(enc_out, axis=3)
        autoenc_out = autoencoder(val)
        draw_batch(val, enc_out_summed, autoenc_out, idx, output_dir, image_prefix)


def draw_batch(val, enc_out_summed, autoenc_out, fig_index, output_dir, image_prefix):
    batch_size = val.shape[0]
    columns = 3

    figure_size = (val.shape[1], val.shape[2])
    fig = plt.figure(figsize=figure_size)

    for row_idx in range(0, batch_size):
        index_base = row_idx * columns + 1

        original_image_idx = index_base
        autoencoder_out_index = index_base + 1
        encoder_out_index = index_base + 2

        # Show original image
        fig.add_subplot(batch_size, columns, original_image_idx)
        plt.imshow(val[row_idx])

        # Show autoencoder output image
        fig.add_subplot(batch_size, columns, autoencoder_out_index)
        plt.imshow(autoenc_out[row_idx])

        # Show encoder output image
        fig.add_subplot(batch_size, columns, encoder_out_index)
        plt.imshow(enc_out_summed[row_idx])
    plt.show()
    fig.savefig(os.path.join(output_dir, f"{image_prefix}_mini_batch_{fig_index}.png"))
    plt.close(fig)


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out",
        help="output directory",
        default=os.path.join(script_path, "out"),
        type=str,
    )
    parser.add_argument(
        "--images",
        help="image_directory",
        type=str,
        default="~/datasets/SPLObjDetectDatasetV2/trainval/images/",
    )
    parser.add_argument(
        "--train_backbone", help="train_backbone", type=bool, default=True
    )
    parser.add_argument("--batch_size", help="batch_size", type=int, default=4)
    parser.add_argument(
        "--gpu_memory_limit",
        help="max memory to be used by each GPU",
        type=int,
        default=7200,
    )
    parser.add_argument("--epochs", help="epochs", type=int, default=20)
    parser.add_argument(
        "--input_shape",
        help="new input shape, (height, width, channels)",
        type=list[int],
        # 4:3 (w:h) aspect ratio;
        default=[100, 130, 1],
    )
    parser.add_argument(
        "--channel_index",
        help="applicable IFF input shape has channel count of 1",
        type=int,
        # 4:3 (w:h) aspect ratio;
        default=0,
    )
    parser.add_argument(
        "--backbone",
        help="backbone path",
        type=str,
        default=os.path.join(script_path, "classifier_backbone.hdf5"),
    )

    args = parser.parse_args("")

    configure_gpu(memory_limit=args.gpu_memory_limit)

    backbone = load_networks(args.backbone)

    new_input_shape, new_batch_input_shape = reshape_backbone_inputs(args.input_shape)

    new_backbone = get_backbone_with_new_shape(
        new_input_shape, new_batch_input_shape, backbone
    )

    new_decoder = get_simple_decoder(new_backbone, 0.25)

    new_autoencoder = get_new_autoencoder(
        args.train_backbone, new_backbone, new_decoder
    )

    tf.keras.models.save_model(
        new_autoencoder, os.path.join(args.out, "autoencoder_untrained_saved_model/")
    )

    regularize_data = True
    dataset = setup_dataset(
        image_shape=new_input_shape,
        image_directory=os.path.expanduser(args.images),
        batch_size=args.batch_size,
        regularize=regularize_data,
        augment=True,
        seed=135
        # channel_index=args.channel_index,
    )

    max_images_to_draw = 40
    batch_count_to_show = int(max_images_to_draw / args.batch_size)

    # show_simple_results(
    #     new_backbone,
    #     new_autoencoder,
    #     dataset,
    #     batch_count_to_show,
    #     args.out,
    #     "pre_train",
    # )

    history = run_training(new_autoencoder, dataset, args.epochs, True)
    new_autoencoder.trainable = False
    new_autoencoder.finalize_state()

    trained_encoder = new_autoencoder.get_layer("encoder")
    trained_encoder.build()
    trained_encoder.compile(**new_autoencoder._get_compile_args())
    trained_encoder.trainable = False
    trained_encoder.finalize_state()

    tf.keras.models.save_model(
        new_autoencoder, os.path.join(args.out, "autoencoder_saved_model/")
    )
    tf.keras.models.save_model(
        new_autoencoder, os.path.join(args.out, "autoencoder.hdf5")
    )
    tf.keras.models.save_model(
        trained_encoder, os.path.join(args.out, "trained_encoder_saved_model/")
    )
    tf.keras.models.save_model(
        trained_encoder, os.path.join(args.out, "trained_encoder.hdf5")
    )
    show_simple_results(
        trained_encoder,
        new_autoencoder,
        dataset,
        batch_count_to_show,
        args.out,
        "trained",
    )


if __name__ == "__main__":
    main()
# %%
