import sys
sys.path.append('../img_ret')

from typing import Union

import torch
from torch import nn

from transformers import AutoImageProcessor, Swinv2Model, Swinv2Config

# Disable Huggingface's warning for models & tokenizers
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


""" This file contains classes for Vision Transformer models to extract image features. """

class BaseEncoder(nn.Module):
    """ Base class for vision models. """
    def __init__(self, model_type:str):
        pass

    def forward(self) -> torch.Tensor:
        """ Forward pass of the model. """
        raise NotImplementedError

    def get_swimv2_model(self, model_size:str) -> Swinv2Model:
        """
        Returns the model object for Swinv2Model Transformer class.

        model_size: The size of the model. Either `tiny` or `base` or `large`
        """
        if model_size == "tiny":
            model_str = "microsoft/swinv2-tiny-patch4-window8-256"
            model = Swinv2Model.from_pretrained(model_str,
                                                output_hidden_states=True)
            image_processor = AutoImageProcessor.from_pretrained(model_str)

        elif model_size == "base":
            # configuration to get untrained weights for the model
            # configuration = Swinv2Config()
            # model = Swinv2Model(configuration)
            # configuration = model.config

            # get the pretrained model weights
            model_str = "microsoft/swinv2-base-patch4-window16-256"
            model = Swinv2Model.from_pretrained(model_str,
                                                output_hidden_states=True)
            image_processor = AutoImageProcessor.from_pretrained(model_str)

        elif model_size == "large":
            model_str = "microsoft/swinv2-large-patch4-window12-192-22k"
            model = Swinv2Model.from_pretrained(model_str,
                                                output_hidden_states=True)
            image_processor = AutoImageProcessor.from_pretrained(model_str)

        else:
            raise ValueError("Invalid model size")

        return model, image_processor


class TransformerEncoder(BaseEncoder):
    """ Inherited class for vision transformer models. """
    def __init__(self, model_type:str, model_size:str):
        super(BaseEncoder, self).__init__()
        self.model_type = model_type
        self.model_size = model_size

        if model_type == "swinv2":
            self.model, self.image_processor = self.get_swimv2_model(model_size)
            self.fc = nn.Linear(768, 512)

        else:
            raise ValueError("Invalid model type")

    def forward(self, *input, **kwargs) -> torch.Tensor:
        """ Forward pass of the model. """
        
        output = self.model(**kwargs)
        print(output)
        
        # Get all the hidden states from the model
        pooler_outout = output['pooler_output']

        # Apply a linear layer to get the final output
        output = self.fc(pooler_outout)
        return output        


if __name__ == "__main__":
    # Test the model
    img_encoder = TransformerEncoder("swinv2", "base")

    # Load the image using PIL
    from PIL import Image
    img_path = '/ds/images/Stanford_Online_Products/bicycle_final/111085122871_0.JPG'

    img = Image.open(img_path)
    image = img_encoder.image_processor(img, return_tensors="pt")

    # Print shape of pooler output
    hidden_states = img_encoder.model(**image)

    