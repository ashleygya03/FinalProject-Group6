"""Model architecture class."""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2

class ASLModel:
    def __init__(self, image_size: int, img_channel: int, n_classes: int):
        self.image_size = image_size
        self.img_channel = img_channel
        self.n_classes = n_classes
        self.model = self._build_model()

    def _build_model(self):
        base_model = MobileNetV2(
            input_shape=(self.image_size, self.image_size, self.img_channel),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = Input(shape=(self.image_size, self.image_size, self.img_channel))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        self.base_model = base_model
        return model

    def get_model(self):
        return self.model

    def unfreeze_top_layers(self, fine_tune_at=100):
        self.base_model.trainable = True
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
