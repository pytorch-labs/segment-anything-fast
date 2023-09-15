import torch
from .segment_anything import sam_model_registry, SamPredictor


def load_predictor(sam_model_type):
    from pathlib import Path
    home = str(Path.home())
    checkpoints = home/"checkpoints"/"sam"
    model_type_to_checkpoint = {
        'vit_h': f'{checkpoints}/sam_vit_h_4b8939.pth',
        'vit_l': f'{checkpoints}/sam_vit_l_0b3195.pth',
        'vit_b': f'{checkpoints}/sam_vit_b_01ec64.pth',
    }

    checkpoint_path = model_type_to_checkpoint[sam_model_type]
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path).cuda()
    return SamPredictor(sam)


VIT_H_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def set_inference(predictor):
    predictor.model.image_encoder = predictor.model.image_encoder.eval().half()
    predictor.model.prompt_encoder = predictor.model.prompt_encoder.eval().half()
    predictor.model.mask_decoder = predictor.model.mask_decoder.eval().half()


LOADED_MODELS = {}


def apply_image_torch(predictor, image: torch.Tensor) -> torch.Tensor:
    transform = predictor.transform
    assert transform.target_length == 1024, "This is an internal failure, please open an issue."
    target_size = transform.get_preprocess_shape(
        image.shape[0], image.shape[1], transform.target_length)
    from torchvision.transforms.functional import resize
    return resize(image, target_size)


def check_image_dim(image):
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() == 4:
        return image
    raise ValueError(
        f"Expected image to be 3 or 4 dimensional, but got {image.dim()} instead.")


def check_image(image, device):
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Expected image to be of type torch.Tensor but got {type(image)} instead.")
    image = check_image_dim(image)
    if image.size(0) == 0:
        raise ValueError(
            f"Expected non-zero batch dimension for 4D input image.")
    if image.device != device:
        raise ValueError(f"Expected image to be on the same device {device} as the loaded model, but got {image.device} instead. ",
                         "You might want to clear the cache and rebuild on the current device or ",
                         "move the input to the same device as the loaded model.")
    return image


def get_image_sizes(image):
    return [i.size()[1:] for i in image.unbind()]


def resize_images(image, device):
    if image.is_nested:
        # TODO: Use torch interpolate with NestedTensor support
        image = torch.nested.nested_tensor(
            list(map(apply_image_torch, image.unbind())))
    else:
        # TODO: Batch this up to work on 4D inputs
        image = torch.stack(list(map(apply_image_torch, image.unbind())))
    return image


def pad_images(image, device):
    if image.is_nested:
        return torch.nested.to_padded_tensor(image, 0, output_size=((image.size(0), image.size(1), 1024, 1024)))
    if not (image.size(-1) == 1024 and image.size(-2) == 1024):
        raise ValueError(
            f"Expected image to be of size 1024x1024, but got {image.size()[2:]} instead.")
    return image


def vit_b_optimized(images, nt_coords):
    if "vit_b_optimized" not in LOADED_MODELS:
        predictor = set_inference(load_predictor("vit_b"))
        from dynamic_quant import apply_dynamic_quant
        apply_dynamic_quant(predictor.model.image_encoder)
        torch.compile(predictor.model.image_encoder, "max-autotune")
        LOADED_MODELS["vit_b_optimized"] = predictor
    predictor = LOADED_MODELS["vit_b_optimized"]
    encoder = predictor.model.image_encoder
    device = predictor.device
    images = check_image(images, device)
    original_sizes = get_image_sizes(images)
    images = resize_images(images)
    input_sizes = get_image_sizes(images)
    images = pad_images(images)
    features_batch = encoder(images)
    nt_fg_labels = torch.nested.nested_tensor([torch.ones(
        (coords.size(0), 1), dtype=torch.int, device=device) for coords in nt_coords.unbind()])
    predictor.reset_image()
    predictor.original_sizes = original_sizes
    predictor.input_sizes = input_sizes
    predictor.features_batch = features_batch
    predictor.is_image_set = True
    nt_coords = nt_coords.unsqueeze(2)
    masks, scores, _ = predictor.predict_torch(
        point_coords=nt_coords,
        point_labels=nt_fg_labels,
        multimask_output=True,
    )
    return masks, scores


def vit_b_approximate(tensor):
    return None


def vit_h_optimized(tensor):
    return None


def vit_h_approximate(tensor):
    return None


def clear_cache():
    LOADED_MODELS.clear()
