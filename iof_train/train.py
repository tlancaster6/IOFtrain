import os
import datetime as dt
import json
from glob import glob
import subprocess as sp

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')

NUMER_OF_TPUS = 1
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results', dt.datetime.now().strftime("%m%d%yT%H%M"))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

with open(os.path.join(PACKAGE_DIR, 'config.json'), 'w') as f:
    config_dict = json.load(f)

TRAIN_DIR = os.path.join(DATA_DIR, config_dict['train_dir'])
VAL_DIR = os.path.join(DATA_DIR, config_dict['val_dir'])

def tfrecord_count(containing_dir):
    """for some reason, loading a tfrecord file through the tflite_model_maker api seems to require that we know, in
    advance, the number of records serialized therein. But the only way (as far as I can tell) to get this info is
    to first deserialize the tfrecord, as seen in this function. This seems inefficient (since the tfrecord is
    presumably deserialized again later) but it works."""
    count = 0
    for tfrecord_path in glob(os.path.join(containing_dir, '*.tfrecord')):
        count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    return count

train_data = object_detector.DataLoader(TRAIN_DIR, tfrecord_count(TRAIN_DIR), ['Fish', 'Pipe'])
val_data = object_detector.DataLoader(VAL_DIR, tfrecord_count(VAL_DIR), ['Fish', 'Pipe'])
spec = model_spec.get(config_dict['model_id'])
model = object_detector.create(train_data, model_spec=spec, batch_size=32, train_whole_model=True, epochs=5,
                               validation_data=val_data, do_train=True)
model.evaluate(val_data)

filename = f'{config_dict["model_id"]}.tflite'
model.export(export_dir=os.path.join(DATA_DIR, 'results'), tflite_filename=filename)
model.evaluate_tflite(filename, val_data)

sp.run(['edgetpu_compiler', filename, f'--num_segments={NUMER_OF_TPUS}'])
edge_filename = f'{config_dict["model_id"]}_edgetpu.tflite'
populator_dst = metadata.MetadataPopulator.with_model_file(edge_filename)

with open(filename, 'rb') as f:
  populator_dst.load_metadata_and_associated_files(f.read())

populator_dst.populate()
updated_model_buf = populator_dst.get_model_buffer()
