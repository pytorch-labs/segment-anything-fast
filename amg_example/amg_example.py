import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    print(f"Saving trace under {path}")
    prof.export_chrome_trace(path)
    return result

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    import torch.utils.benchmark as benchmark
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image = cv2.imread('dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def run(sam_model_registry, SamAutomaticMaskGenerator, optimize=False):
    sam_checkpoint = "/home/cpuhrsch/saf/experiments_data/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    if optimize:
        # Apply optimizations
        from segment_anything_fast.tools import apply_eval_dtype_predictor
        mask_generator.predictor = apply_eval_dtype_predictor(mask_generator.predictor, torch.bfloat16)
        mask_generator.predictor.model.image_encoder = torch.compile(mask_generator.predictor.model.image_encoder, mode="max-autotune")
    masks = mask_generator.generate(image)
    ms = benchmark_torch_function_in_microseconds(mask_generator.generate, image)
    print(f"Generating this mask takes {ms}ms. We set optimize to {optimize}.")
    profiler_runner(f"asdf_{optimize}.json.gz", mask_generator.generate, image)
    plt.figure(figsize=(image.shape[1]/100., image.shape[0]/100.), dpi=100)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.tight_layout()
    if optimize:
        plt.savefig('dog_mask_fast.png', format='png')
    else:
        plt.savefig('dog_mask.png', format='png')

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
run(sam_model_registry, SamAutomaticMaskGenerator)

from segment_anything_fast import sam_model_registry, SamAutomaticMaskGenerator
run(sam_model_registry, SamAutomaticMaskGenerator, optimize=True)
