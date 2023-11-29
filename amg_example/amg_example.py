import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch.utils.benchmark as benchmark

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    print(f"Saving trace under {path}")
    prof.export_chrome_trace(path)
    return result

def benchmark_torch_function_in_milliseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e3

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

def save_masks(masks, filename):
    plt.figure(figsize=(image.shape[1]/100., image.shape[0]/100.), dpi=100)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{filename}.png', format='png')

image = cv2.imread('dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


from segment_anything_fast import sam_model_registry, sam_model_fast_registry, SamAutomaticMaskGenerator

sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
baseline = True

sam = sam_model_fast_registry[model_type](checkpoint=sam_checkpoint, compile_mode='default')
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
import pdb; pdb.set_trace()
save_masks(masks, 'dog_mask_fast_fast')
print(f"fast: {benchmark_torch_function_in_milliseconds(mask_generator.generate, image)}ms")
profiler_runner(f"asdf_True.json.gz", mask_generator.generate, image)
