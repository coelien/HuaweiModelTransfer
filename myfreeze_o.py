import tensorflow as tf
import importlib
import os
import numpy as np
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# # 先加载图和参数变量
# saver = tf.train.import_meta_graph('./checkpoints/bsrn_c64_s64_x234.ckpt.meta')
# with tf.Session() as sess:
#     saver.restore(sess, './checkpoints/bsrn_c64_s64_x234.ckpt')

def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("sr_input:0")
            input_image_scale = sess.graph.get_tensor_by_name("sr_input_scale:0")

            # output_list = [sess.graph.get_tensor_by_name("generator/add:0")]
            # for i in range(1, 16):
            #     op = sess.graph.get_tensor_by_name("generator/add_{}:0".format(str(i)))
            #     output_list.append(op)

            # output = tf.concat(output_list, axis=0, name="output")
            output = sess.graph.get_tensor_by_name("output:0")

            # 读取测试图片,此处使用假数据
            im = np.random.randn(1, 60, 60, 3)

            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字

            out = sess.run(output, feed_dict={input_image_tensor: im,
                                              input_image_scale: 4})
            out1 = tf.split(out, 16, axis=0)
            print("out:{}".format(out))
            print(im.shape)

            for i in out1:
                print(i.shape)


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "output"
    # for i in range(1, 16):
    #     output_node_names += ",generator/add_{}".format(i)

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_list = [sess.graph.get_tensor_by_name("generator/add:0")]
        for i in range(1, 16):
            op = sess.graph.get_tensor_by_name("generator/add_{}:0".format(str(i)))
            output_list.append(op)

        _ = tf.concat(output_list, axis=0, name="output")
        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint = '../temp/results/model.ckpt-1000000'
    # 输出pb模型的路径
    out_pb_path = "./frozen_model.pb"
    # 调用freeze_graph将ckpt转为pb
    freeze_graph(input_checkpoint, out_pb_path)

    # # 测试pb模型
    image_path = ''
    freeze_graph_test(pb_path=out_pb_path, image_path=image_path)