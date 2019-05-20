# crnn_self_attetion

使用self attention 实现的crnn网络，效果比直接使用seq2seq + attention好了很多，在长序列，多单词情况下效果也很好

使用方法更改 dataset/write_tfrecords.py 文件中路径位自己的路径，同样使用的是 mjsynth.tar.gz http://www.robots.ox.ac.uk/~vgg/data/text/ 数据集

然后更改 dataset/data_provider.py 中 TfDataProvider 类中的root路径为生成的tfrecord路径

运行 crnn_main.py 训练
运行 evaluate_net.py 测试
