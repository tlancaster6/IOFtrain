import os
import datetime as dt
import subprocess as sp
from shutil import move
import definitions as defs
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
import tensorflow as tf
import wandb
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')


def train(data_dir, model_id, epochs=10, batch_size=16, use_wandb=True):
    run_name = dt.datetime.now().strftime("%m%d%yT%H%M")
    results_dir = os.path.join(defs.RESULTS_DIR, run_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f'resuts will be saved to {results_dir}')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    label_map = {1: 'fish', 2: 'pipe'}
    train_data = object_detector.DataLoader.from_pascal_voc(train_dir, train_dir, label_map=label_map)
    val_data = object_detector.DataLoader.from_pascal_voc(val_dir, val_dir, label_map=label_map)

    info_log = open(os.path.join(results_dir, 'info.log'), 'w')
    info_log.write(f'n_train: {len(train_data)}\n')
    info_log.write(f'n_validL {len(val_data)}\n')
    info_log.write(f'epochs: {epochs}\n')
    info_log.write(f'model_id: {model_id}\n')
    info_log.write(f'batch_size{batch_size}\n')
    info_log.close()

    my_model_spec = model_spec.get(model_id)
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    my_model_spec.config.model_dir = checkpoint_dir
    my_model_spec.config.map_freq = 1
    my_model_spec.config.num_classes = 2
    my_model_spec.config.label_map = label_map
    my_model_spec.config.num_epochs = epochs
    my_model_spec.config.batch_size = batch_size
    if use_wandb:
        wandb.init(project='IOFtrain', sync_tensorboard=True, reinit=True)
        wandb.run.name = run_name
    model = object_detector.create(train_data, model_spec=my_model_spec, batch_size=batch_size, train_whole_model=True,
                                   epochs=epochs, validation_data=val_data, do_train=True)

    eval_log = open(os.path.join(results_dir, 'evaluation.log'), 'w')
    eval_log.write('prequant evaluation:\n')
    prequant_eval = model.evaluate(val_data)
    eval_log.write(f'{prequant_eval}\n')

    tflite_filename = f'{model_id}.tflite'
    labels_filename = 'labels.txt'
    model.export(export_dir=results_dir, tflite_filename=tflite_filename, label_filename=labels_filename,
                 export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
    eval_log.write('\npostquant evaluation\n')
    postquant_eval = model.evaluate_tflite(os.path.join(results_dir, tflite_filename), val_data)
    eval_log.write(f'{postquant_eval}\n')

    sp.run(['edgetpu_compiler', os.path.join(results_dir, tflite_filename), f'--num_segments={defs.NUMER_OF_TPUS}'])
    edge_filename = tflite_filename.split('.')[0] + '_edgetpu.tflite'
    log_filename = tflite_filename.split('.')[0] + '_edgetpu.log'
    move(os.path.join(defs.BASE_DIR, edge_filename), results_dir)
    move(os.path.join(defs.BASE_DIR, log_filename), results_dir)
    eval_log.close()

    if use_wandb:
        wandb.config.update(my_model_spec.config.as_dict())
        wandb.config.update({'data_source': dir.split('/')[-1]})
        for key, val in prequant_eval.items():
            wandb.run.summary[f'prequant_{key}'] = val
        for key, val in postquant_eval.items():
            wandb.run.summary[f'postquant_{key}'] = val
        wandb.finish()
