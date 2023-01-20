import os
import datetime as dt
import subprocess as sp
from shutil import move, copy
import definitions as defs
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')

def train(data_dir, model_id):
    results_dir = os.path.join(defs.RESULTS_DIR, dt.datetime.now().strftime("%m%d%yT%H%M"))
    print(f'resuts will be saved to {results_dir}')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    label_map = {1: 'fish', 2: 'pipe'}
    train_data = object_detector.DataLoader.from_pascal_voc(train_dir, train_dir, label_map=label_map)
    val_data = object_detector.DataLoader.from_pascal_voc(val_dir, val_dir, label_map=label_map)

    my_model_spec = model_spec.get(model_id)
    model = object_detector.create(train_data, model_spec=my_model_spec, batch_size=32, train_whole_model=True, epochs=100,
                                   validation_data=val_data, do_train=True)
    eval_log = open(os.path.join(results_dir, 'evaluation.log'), 'w')
    print('prequant evaluation:', file=eval_log)
    print(model.evaluate(val_data), file=eval_log)

    tflite_filename = f'{model_id}.tflite'
    labels_filename = 'labels.txt'
    model.export(export_dir=results_dir, tflite_filename=tflite_filename, label_filename=labels_filename,
                 export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
    print('\npostquant evaluation', file=eval_log)
    print(model.evaluate_tflite(os.path.join(results_dir, tflite_filename), val_data), file=eval_log)

    sp.run(['edgetpu_compiler', os.path.join(results_dir, tflite_filename), f'--num_segments={defs.NUMER_OF_TPUS}'])
    edge_filename = tflite_filename.split('.')[0] + '_edgetpu.tflite'
    log_filename = tflite_filename.split('.')[0] + '_edgetpu.log'
    move(os.path.join(defs.BASE_DIR, edge_filename), results_dir)
    move(os.path.join(defs.BASE_DIR, log_filename), results_dir)
    eval_log.close()