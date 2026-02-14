from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import to_numpy_array

from ..base import BaseImageProcessor, InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class ImageProcessor(BaseImageProcessor):
    """Image processor for LLaVA models.

    mlx-vlm's training pipeline expects a BaseImageProcessor so that it can
    deterministically build MLX pixel_values and expand <image> placeholders.
    """

    def __init__(self, config=None, **kwargs):
        # Prefer model config image size when available (e.g. 336 for LLaVA-1.5)
        if config is not None:
            try:
                img_size = (
                    (config.get("vision_config", {}) or {}).get("image_size", None)
                )
                if img_size is not None:
                    kwargs.setdefault("size", (int(img_size), int(img_size)))
                    kwargs.setdefault(
                        "crop_size",
                        {"height": int(img_size), "width": int(img_size)},
                    )
            except Exception:
                pass
        super().__init__(**kwargs)

    def preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        from functools import partial, reduce

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(
                resize,
                size=self.size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        return images


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        if isinstance(self.vision_feature_layer, int):
            if self.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]

        else:
            hs_pool = [
                hidden_states[layer_idx] for layer_idx in self.vision_feature_layer
            ]
            # For default; crop CLS from each hidden state in the hidden state pool
            if self.vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = mx.concatenate(hs_pool, axis=-1)

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids.
        # Support batching: image_features is (B, T_img, H) and input_ids is (B, S).
        batch_size, t_img, vision_hidden_size = image_features.shape

        # cast to the dtype of the input_embeds to support quantized models
        image_features = image_features.astype(inputs_embeds.dtype)

        # For each sample in the batch, replace the <image> token positions with
        # the projected image features.
        for b in range(batch_size):
            # input_ids[b] is (S,). np.where returns tuple with one array.
            image_positions = np.where(np.array(input_ids[b]) == image_token_index)[0]

            if len(image_positions) != t_img:
                raise ValueError(
                    f"Llava expected {t_img} <image> tokens for sample {b}, got {len(image_positions)}. "
                    "This usually means the text prompt wasn't expanded to match the number of image tokens."
                )

            inputs_embeds[b, image_positions.tolist(), :] = image_features[b]

        return inputs_embeds

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            input_ids,
            mask=mask,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits
