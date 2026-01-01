import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
from IPython.display import HTML
from pipeline import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video


if __name__ == "__main__":


    # variant='diffusion_transformer_384p'       # For low resolution variant
    variant='diffusion_transformer_768p'     # For high resolution variant

    model_name = "pyramid_flux"   # select the model "pyramid_flux" or "pyramid_mmdit"

    model_path = "/content/drive/MyDrive/PATH/pyramid-flow-miniflux"   # The downloaded checkpoint dir
    # model_path = "/path/to/your/finetuned/output_folder"
    model_dtype = 'bf16'

    device_id = 0
    torch.cuda.set_device(device_id)

    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_name=model_name,
        model_variant=variant,
    )

    model.vae.to("cuda")
    target_module = model.dit
    model.text_encoder.to("cuda")
    lora_path = "/content/drive/MyDrive/DiT_Checkpoint/checkpoint-2000"

    
    # If you used PEFT/Diffusers to train
    from peft import PeftModel
    target_module = PeftModel.from_pretrained(target_module, lora_path)
    # Merge weights (optional, makes inference faster)
    # target_module.merge_and_unload()
    print(">>>>>>>>>>>>>>>>>>>>>", target_module)

    












    