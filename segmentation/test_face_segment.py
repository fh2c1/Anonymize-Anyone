import cv2
import json
import numpy as np
from pathlib import Path

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.projects import point_rend

def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

def get_mask(outputs, class_dict):
    mask_image = np.zeros((*outputs['instances'].image_size, 3), np.uint8)
    class_array = outputs['instances'].pred_classes.cpu().numpy()

    draw_order = sorted(class_dict.keys(), key=int)
    for class_id in draw_order:
        indices = np.where(class_array == int(class_id))[0]
        for i in indices:
            mask_image[outputs['instances'].pred_masks[i].cpu().numpy()] = class_dict[class_id][1]

    return mask_image

def setup_cfg(num_of_class):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file('./InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml')
    cfg.MODEL.WEIGHTS = './segmentation/pointrend/model_final.pth'
    cfg.MODEL.DEVICE = 'cuda:0'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_of_class
    return cfg

def process_image(im_name, predictor, class_dict_RGB, im_input_path, im_output_path):
    imageName = f'{im_input_path}/{im_name}'
    im = cv2.imread(imageName)
    print(im_name)

    outputs = predictor(im)
    im_mask = get_mask(outputs, class_dict_RGB)
    cv2.imwrite(f'{im_output_path}/{im_name}', im_mask)

def main():
    register_coco_instances(
        'my_dataset',
        {},
        './segmentation/dataset/_annotations.coco.json',
        './segmentation/dataset',
    )

    with open('./segmentation/dataset/_annotations.coco.json', 'r') as json_file:
        json_data = json.load(json_file)
        num_of_class = len(json_data['categories'])
        print('num_of_class: ', num_of_class)

    cfg = setup_cfg(num_of_class)
    predictor = DefaultPredictor(cfg)

    class_dict_RGB = {
        '10': ['face', hex_to_rgb('#44690C')],
        '6': ['nose', hex_to_rgb('#569A93')],
        '7': ['upper_lip', hex_to_rgb('#A6480A')],
        '8': ['lower_lip', hex_to_rgb('#F25F41')],
        '0': ['eye_brow', hex_to_rgb('#F19BDC')],
        '4': ['LWA', hex_to_rgb('#0000FF')],
        '5': ['M-C', hex_to_rgb('#DE9846')],
        '3': ['caruncle', hex_to_rgb('#F4F812')],
        '2': ['iris', hex_to_rgb('#805472')],
        '1': ['double_eyelid', hex_to_rgb('#D60ACF')],
        '9': ['maker', hex_to_rgb('#7C3E3D')],
        '11': ['double_eyelid_2', hex_to_rgb('#ff7f00')],
        '12': ['double_eyelid_3', hex_to_rgb('#ffff00')],
        '13': ['double_eyelid_4', hex_to_rgb('#008000')],
        '14': ['double_eyelid_5', hex_to_rgb('#0067a3')]
    }

    im_input_path = './segmentation/input'
    im_list = ['output_1.png', 'output_2.png']
    im_output_path = './segmentation/output'
    Path(im_output_path).mkdir(exist_ok=True, parents=True)

    for im_name in im_list:
        process_image(im_name, predictor, class_dict_RGB, im_input_path, im_output_path)

if __name__ == '__main__':
    main()