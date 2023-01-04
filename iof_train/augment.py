import albumentations as A
import cv2
import xmltodict
import pascal_voc_writer
from glob import glob
import os

transform1 = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2),
)

def read_voc_xml(file_path):
    with open(file_path, 'r') as f:
        xml_content = f.read()
        xml_dict = xmltodict.parse(xml_content)['annotation']
        filename = xml_dict['filename']
        objects = xml_dict['object']
        bboxes = [[int(o['bndbox'][key]) for key in ['xmin', 'ymin', 'xmax', 'ymax']] for o in objects]

        return bboxes, filename

def get_xml_paths(source_dir):
    xml_paths = glob(os.path.join(source_dir, '*.xml'))
    return xml_paths

def apply_augmentation(source_xml_path, transform_func):
    bboxes, fname = read_voc_xml(source_xml_path)
    img_path = os.path.join(os.path.dirname(source_xml_path), fname)
    source_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    transformed = transform_func(image=source_img, bboxes=bboxes)
    return transformed, fname

def apply_augmentations(source_dir, transform_func, output_dir, reps_per_im):
    xml_paths = get_xml_paths(source_dir)
    for xml_p in xml_paths:
        for rep in range(reps_per_im):
            transformed, fname,  = apply_augmentation(xml_p, transform_func)
            img_out_path = os.path.join(output_dir, f'{os.path.splitext(fname)}_aug{rep}.jpg')
            cv2.imwrite(img_out_path, transformed['image'])
            xml_out_path = img_out_path.replace('.jpg', '.xml')
            writer = pascal_voc_writer.Writer(img_out_path, )








