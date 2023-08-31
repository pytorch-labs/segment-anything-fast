import torch
from pycocotools.coco import COCO
import numpy as np
from scipy import ndimage
import skimage.io as io
import skimage.color as color


def _get_center_point(mask, ann_id, cache):
    """
    This is a rudimentary version of https://arxiv.org/pdf/2304.02643.pdf,
    section D.1.Point Sampling

    From the paper: "The first point is chosen deterministically as the point
    farthest from the object boundary."

    The code below is an approximation of this.

    First, we try to calculate the center of mass. If it's inside the mask, we
    stop here.

    The centroid may be outside of the mask for some mask shapes. In this case
    we do a slow hack, specifically, we check for the
    minumum of the maximum distance from the boundary in four directions
    (up, right, down, left), and take the point with the maximum of these
    minimums. Note: this is not performant for large masks.

    Returns the center point in (x, y) format
    """
    if ann_id in cache:
        return cache[ann_id]

    # try the center of mass, keep it if it's inside the mask
    com_y, com_x = ndimage.center_of_mass(mask)
    com_y, com_x = int(round(com_y, 0)), int(round(com_x, 0))
    if mask[com_y][com_x]:
        cache[ann_id] = (com_x, com_y)
        return (com_x, com_y)

    # if center of mass didn't work, do the slow manual approximation

    # up, right, down, left
    # TODO(future): approximate better by adding more directions
    distances_to_check_deg = [0, 90, 180, 270]

    global_min_max_distance = float('-inf')
    global_coords = None
    # For now, terminate early to speed up the calculation as long as
    # the point sample is gooe enough. This sacrifices the quality of point
    # sampling for speed. In the future we can make this more accurate.
    DISTANCE_GOOD_ENOUGH_THRESHOLD = 20

    # Note: precalculating the bounding box could be somewhat
    #   helpful, but checked the performance gain and it's not much
    #   so leaving it out to keep the code simple.
    # Note: tried binary search instead of incrementing by one to
    #   travel up/right/left/down, but that does not handle masks
    #   with all shapes properly (there could be multiple boundaries).
    for row_idx in range(mask.shape[0]):
        for col_idx in range(mask.shape[1]):
            cur_point = mask[row_idx, col_idx]

            # skip points inside bounding box but outside mask
            if not cur_point:
                continue

            max_distances = []
            for direction in distances_to_check_deg:
                # TODO(future) binary search instead of brute forcing it if we
                # need a speedup, with the cache it doesn't really matter though
                if direction == 0:
                    # UP
                    cur_row_idx = row_idx

                    while cur_row_idx >= 0 and mask[cur_row_idx, col_idx]:
                        cur_row_idx = cur_row_idx - 1
                    cur_row_idx += 1
                    distance = row_idx - cur_row_idx
                    max_distances.append(distance)

                elif direction == 90:
                    # RIGHT
                    cur_col_idx = col_idx

                    while cur_col_idx <= mask.shape[1] - 1 and \
                            mask[row_idx, cur_col_idx]:
                        cur_col_idx += 1
                    cur_col_idx -= 1
                    distance = cur_col_idx - col_idx
                    max_distances.append(distance)

                elif direction == 180:
                    # DOWN
                    cur_row_idx = row_idx
                    while cur_row_idx <= mask.shape[0] - 1 and \
                            mask[cur_row_idx, col_idx]:
                        cur_row_idx = cur_row_idx + 1
                    cur_row_idx -= 1
                    distance = cur_row_idx - row_idx
                    max_distances.append(distance)

                elif direction == 270:
                    # LEFT
                    cur_col_idx = col_idx
                    while cur_col_idx >= 0 and mask[row_idx, cur_col_idx]:
                        cur_col_idx -= 1
                    cur_col_idx += 1
                    distance = col_idx - cur_col_idx
                    max_distances.append(distance)

            min_max_distance = min(max_distances)
            if min_max_distance > global_min_max_distance:
                global_min_max_distance = min_max_distance
                global_coords = (col_idx, row_idx)
            if global_min_max_distance >= DISTANCE_GOOD_ENOUGH_THRESHOLD:
                break

    cache[ann_id] = global_coords
    return global_coords


def build_datapoint(imgId, coco, pixel_mean, pixel_std, coco_root_dir, coco_slice_name, catIds, cache, predictor):
    img = coco.loadImgs(imgId)[0]

    file_location = f'{coco_root_dir}/{coco_slice_name}/{img["file_name"]}'
    I = io.imread(file_location)
    if len(I.shape) == 2:
        # some images, like img_id==61418, are grayscale
        # convert to RGB to ensure the rest of the pipeline works
        I = color.gray2rgb(I)

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # approximate the center point of each mask
    coords_list = []
    gt_masks_list = []
    for ann in anns:
        ann_id = ann['id']
        mask = coco.annToMask(ann)
        gt_masks_list.append(torch.tensor(mask))
        coords = _get_center_point(mask, ann_id, cache)
        coords_list.append(coords)

    image = I

    # predictor_set_image begin
    # Transform the image to the form expected by the model
    input_image = predictor.transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image)
    input_image_torch = input_image_torch.permute(
        2, 0, 1).contiguous()[None, :, :, :]
    predictor_input_size = input_image_torch.shape[-2:]

    # Preprocess
    x = input_image_torch
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = predictor.model.image_encoder.img_size - h
    padw = predictor.model.image_encoder.img_size - w
    x = torch.nn.functional.pad(x, (0, padw, 0, padh))

    gt_masks_list = torch.stack(gt_masks_list) if len(gt_masks_list) else None
    return image, coords_list, gt_masks_list, anns, x, predictor_input_size


def build_data(coco_img_ids, coco, catIds, coco_root_dir, coco_slice_name, cache, predictor, use_half, use_half_decoder):

    pixel_mean = predictor.model.pixel_mean.cpu()
    pixel_std = predictor.model.pixel_std.cpu()

    def build_batch(indicies):
        batch = [[], [], [], [], [], [], [], [], [], [], []]
        batch[8] = [0]
        batch[10] = [0]
        for img_idx in indicies:
            imgId = coco_img_ids[img_idx]

            I, coords_list, gt_masks_list, anns, x, predictor_input_size = build_datapoint(imgId,
                                                                                           coco,
                                                                                           pixel_mean,
                                                                                           pixel_std,
                                                                                           coco_root_dir,
                                                                                           coco_slice_name,
                                                                                           catIds,
                                                                                           cache,
                                                                                           predictor)
            if len(coords_list) == 0:
                continue
            batch[0].append(I)
            coords_list = predictor.transform.apply_coords(
                np.array(coords_list), I.shape[:2])
            coords_list = torch.tensor(coords_list, dtype=torch.float)
            batch[1].append(coords_list.reshape(-1))
            batch[2].append(coords_list.size())
            batch[3].append(gt_masks_list.reshape(-1))
            batch[4].append(anns)
            batch[5].append(x)
            batch[6].append(predictor_input_size)
            batch[7].append(img_idx)
            batch[8].append(coords_list.numel() + batch[8][-1])
            batch[9].append(gt_masks_list.size())
            batch[10].append(gt_masks_list.numel() + batch[10][-1])
        if use_half_decoder:
            batch[1] = torch.cat(batch[1]).half() if len(
                batch[0]) > 0 else None
        else:
            batch[1] = torch.cat(batch[1]) if len(batch[0]) > 0 else None
        batch[3] = torch.cat(batch[3]) if len(batch[0]) > 0 else None
        if use_half:
            batch[5] = torch.cat(batch[5]).half() if len(
                batch[0]) > 0 else None
        else:
            batch[5] = torch.cat(batch[5]) if len(batch[0]) > 0 else None
        return batch

    return build_batch


def setup_coco_img_ids(coco_root_dir, coco_slice_name, coco_category_names, img_id):
    annFile = '{}/annotations/instances_{}.json'.format(
        coco_root_dir, coco_slice_name)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_cat = {cat['id']: cat for cat in cats}
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    if coco_category_names is not None:
        catIds = coco.getCatIds(catNms=coco_category_names)
    else:
        catIds = coco.getCatIds()

    if img_id is not None:
        coco_img_ids = [img_id]
    elif coco_category_names is None:
        coco_img_ids = coco.getImgIds()
    else:
        coco_img_ids = coco.getImgIds(catIds=catIds)

    return coco_img_ids, cat_id_to_cat, catIds, coco
