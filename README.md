# Simple Snoring Detection
## 高级机器学习大作业4-1

## 数据准备
先创建目录`./snoring_dataset`和`./test_data`，接着下载数据集[此页面](https://www.kaggle.com/datasets/tareqkhanemu/snoring)，并放在目录`./snoring_dataset`下(呈现成`./snoring_dataset/0`的形式)，并自行准备测试要用的wav格式音频`example.wav`并放在`./test_data`目录下

## 运行
```run
python train.py
python main.py --audio ./test_data/example.wav --model ./checkpoints/snore_net.pth
```
## 引用
如果使用该项目，请标明出处

感谢！！
