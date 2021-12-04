from tensorflow.python.tools import freeze_graph
import argparse
import importlib
import tensorflow as tf
# 导入网络模型文件
import dataloaders
import models

DEFAULT_DATALOADER = 'basic_loader'
DEFAULT_MODEL = 'bsrn'

FLAGS = tf.flags.FLAGS



if __name__ == '__main__':
  tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_string('scales', '2,3,4', 'Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')

  tf.flags.DEFINE_string('restore_path', None, 'Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
  tf.flags.DEFINE_integer('restore_global_step', 0, 'Global step of the restored model. Some models may require to specify this.')
  tf.flags.DEFINE_string('obs_dir', "obs://bsrn-test/", "obs result path, not need on gpu and apulis platform")
  tf.flags.DEFINE_string('train_path', './train/', 'Base path of the trained model to be saved.')
  tf.flags.DEFINE_boolean('ensemble_only', False, 'Calculate (and save) ensembled image only.')
  tf.flags.DEFINE_string('save_path', None, 'Base path of the upscaled images. Specify this to save the upscaled images.')

  tf.flags.DEFINE_string("chip", "gpu", "Run on which chip, (npu or gpu or cpu)")
  tf.flags.DEFINE_string("platform", "linux", 'the platform this code is running on')

  # parse data loader and model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--dataloader', default=DEFAULT_DATALOADER)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.dataloader is not None):
    DATALOADER_MODULE = importlib.import_module('dataloaders.' + pre_parsed.dataloader)
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


if FLAGS.chip == 'npu':
  from npu_bridge.npu_init import *

def main():
    tf.reset_default_graph()
    # initialize
    FLAGS.bsrn_intermediate_outputs = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
    tf.logging.set_verbosity(tf.logging.INFO)

    # data loader
    dataloader = DATALOADER_MODULE.create_loader()
    dataloader.prepare()

    # model
    model = MODEL_MODULE.create_model()
    model.prepare(is_training=False, global_step=FLAGS.restore_global_step)

    # model > restore
    model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
    tf.logging.info('restored the model')
    #
    # inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    # # 定义网络的输出节点
    # # tf_output = model._generator(input_list=inputs, num_modules=FLAGS.bsrn_recursions, scale=model.tf_scale, intermediate_outputs=FLAGS.bsrn_intermediate_outputs, recursion_frequency=FLAGS.bsrn_recursion_frequency, reuse=False)
    # output = tf.identity(model.tf_output, name="output")
    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.io.write_graph(sess.graph_def, FLAGS.save_path, 'model.pb')
        # tf.train.write_graph(sess.graph_def, FLAGS.save_path, 'model.pb')  # 通过write_graph生成模型文件
        # freeze_graph.freeze_graph(
        #     input_graph=FLAGS.save_path + '/model.pb',  # 传入write_graph生成的模型文件
        #     input_saver='',
        #     input_binary=False,
        #     input_checkpoint=FLAGS.restore_path,  # 传入训练生成的checkpoint文件
        #     output_node_names='output',  # 与定义的推理网络输出节点保持一致
        #     restore_op_name='save/restore_all',
        #     filename_tensor_name='save/Const:0',
        #     output_graph=FLAGS.save_path + '/bsrn.pb',  # 改为需要生成的推理网络的名称
        #     clear_devices=False,
        #     initializer_nodes='')
    print("done")

    if FLAGS.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs
        modelarts_result2obs(FLAGS)

if __name__ == '__main__':
    main()