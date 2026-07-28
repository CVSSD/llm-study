# Introduction
Merchaine learning is designed to process problems without a precise stander.  
For example, weather predict, costumn behaviour predict.  
This kind of problems can't be solved by a hardcoded code.  
Mechaine learning can give a answer with a input.  
The input can be text, image, audio...  
Mechaine learning will use a serious of parameters with some topology to process the input data and gerenate a answer.  
Such a set of parameter is called model.  
A data set is need to train the parameters.  
The data set include feature and label. Feature is the data input.  
Label is what this input will be categoried into.  

The mechaine include components data, model, objective function, optimization algorithms.  
Objection function evaluate the result of the trained model. Usually set to lower is better. Thus also called loss function.  

Supervised learning verse Unsupervised learning  
The supervised learning aim at verious specific tasks, classification, regression, tagging, sequence learning.  
The sequence learning is very important concept. I kind of think it is highly related to following advenced algorithm.  
The input and output will be various length sequence.   
For example, voice recorgnition will need to convert a unknown length input samples to a unknown length of text.  
  
Unsupervised learning discovers patterns or structures in unlabeled data.  
The unsupervised learning can process with unlabled data.  
### Improved version
Machine learning is a way of solving problems where it is difficult to manually write explicit rules. Instead of programming the solution, we train a model to learn patterns from data. The inputs can be images, text, audio, or numerical data, and the model predicts the desired outputs.  
  
A machine learning model consists of an architecture and a set of learnable parameters. During training, these parameters are adjusted using a dataset.  
  
In supervised learning, the dataset contains features (inputs) and labels (desired outputs). The goal is to learn a mapping from features to labels.  
  
A machine learning system has four main components:  
  
Data – training examples.  
Model – a mathematical function with learnable parameters.  
Loss (objective) function – measures how well the model performs.  
Optimization algorithm – updates the parameters to minimize the loss.  
  
Supervised learning includes tasks such as classification, regression, tagging, and sequence learning. Sequence learning is especially important because both the input and output may have variable lengths, such as in speech recognition, machine translation, and large language models.  
   
Unsupervised learning uses unlabeled data to discover hidden patterns or structures, such as clustering and dimensionality reduction.  

# Linera algebra
The linear algebra is the basic of the mechaine learning.    
机器学习通过训练一组权重实现数据A到数据B的变化，从集合A到集合B的映射。    
以上概念和矩阵运算的概念相符。矩阵运算是进行机器学习的基础。    
同时矩阵运算可以将单次运算拆解为同时运行的多次计算。因此天生适配GPU。   
CORRECT VERSION：  
机器学习的目标是学习一个从输入数据集合 A 到输出数据集合 B 的映射函数。  
在监督学习中，我们通常利用大量样本训练模型，使模型自动学习一组参数（权重 Weight 和偏置 Bias），从而逼近这个映射关系。  
对于线性层而言，这种映射可以表示为  
y=Wx+b  
其中权重矩阵 W 决定了输入特征如何被线性变换到新的特征空间，因此矩阵本质上就是线性变换的数学表示。  
多层神经网络则通过不断重复"矩阵变换 + 非线性激活"，逐步学习越来越复杂的数据映射，而不仅仅是简单的线性关系  

矩阵乘法由大量彼此独立的乘法与加法组成，不同元素之间通常可以并行计算，因此非常适合 GPU 的 SIMD/SIMT 并行计算架构。GPU 能够同时调度成千上万个计算核心完成这些运算，从而显著提高神经网络训练和推理速度。  

