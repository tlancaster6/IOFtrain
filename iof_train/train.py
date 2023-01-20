import os
import datetime as dt
import json
import subprocess as sp
from shutil import move, copy

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')

NUMER_OF_TPUS = 1
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', dt.datetime.now().strftime("%m%d%yT%H%M"))
CHECKPNT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
if not os.path.exists(CHECKPNT_DIR):
    os.makedirs(CHECKPNT_DIR)

with open(os.path.join(PACKAGE_DIR, 'config.json'), 'r') as f:
    config_dict = json.load(f)
copy(os.path.join(PACKAGE_DIR, 'config.json'), RESULTS_DIR)

TRAIN_DIR = os.path.join(DATA_DIR, config_dict['train_dir'])
VAL_DIR = os.path.join(DATA_DIR, config_dict['val_dir'])

label_map = {1: 'fish'}
train_data = object_detector.DataLoader.from_pascal_voc(TRAIN_DIR, TRAIN_DIR, label_map=label_map)
val_data = object_detector.DataLoader.from_pascal_voc(VAL_DIR, VAL_DIR, label_map=label_map)

mod_spec = model_spec.get(config_dict['model_id'])
mod_spec.config.optimizer = 'adam'
# mod_spec.config.input_rand_hflip = False
mod_spec.config.model_dir = CHECKPNT_DIR
mod_spec.config.map_freq = 1


model = object_detector.create(train_data, model_spec=mod_spec, batch_size=16, train_whole_model=True, epochs=1000,
                               validation_data=val_data, do_train=True)
eval_log = open(os.path.join(RESULTS_DIR, 'evaluation.log'), 'w')
print('prequant evaluation:', file=eval_log)
print(model.evaluate(val_data), file=eval_log)

TFLITE_FILENAME = f'{config_dict["model_id"]}.tflite'
LABELS_FILENAME = 'labels.txt'
model.export(export_dir=RESULTS_DIR, tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
print('\npostquant evaluation', file=eval_log)
print(model.evaluate_tflite(os.path.join(RESULTS_DIR, TFLITE_FILENAME), val_data), file=eval_log)

sp.run(['edgetpu_compiler', os.path.join(RESULTS_DIR, TFLITE_FILENAME), f'--num_segments={NUMER_OF_TPUS}'])
EDGE_FILENAME = TFLITE_FILENAME.split('.')[0] + '_edgetpu.tflite'
LOG_FILENAME = TFLITE_FILENAME.split('.')[0] + '_edgetpu.log'
move(os.path.join(BASE_DIR, EDGE_FILENAME), RESULTS_DIR)
move(os.path.join(BASE_DIR, LOG_FILENAME), RESULTS_DIR)
eval_log.close()