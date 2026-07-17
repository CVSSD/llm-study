Day1
    Finish the enviroment setup. Create the llm-study envirement with miniconda. All necessary packege installed.
    Build up file stuecture for week0.

Day 2 
todo:
1. build a experiment to reduce the memory use of a modle.
Test with a set of random parameters. To invest how the size of parameter set is reduce. 
Load FP32 model
        ↓
Too much VRAM
        ↓
Convert to FP16
        ↓
Half memory
        ↓
Quantize to INT8/INT4
        ↓
Fits on laptop GPU

2.[DONE] Do a linear algebra warn up with tensor. 

LOG:
Make up a plan to got through torch baisc and deep learning theory.
Using https://github.com/pytorch/tutorials and https://github.com/d2l-ai/d2l-en
Now finished d2l cp1 introductioni.

Day 3
Got to chapter 2preliminaries. Finished ndarray, linear algebra and calculus.  
The knowledge in book about linera algebra ,Derivatives and Gradients is just warm up.  
Go deeper in there topics later is necessary.  
Will gome back to this topic when meet difficult in later chapters.

When doing the linera algebra warm up, found the data strecture maters a lot for large scale matrix calculation.
It is necessary to know the tensor memory archiecture.

Day 3
Go through the math concepts, deverative, limitation, gradient, matrix operation.
Tyler expension. 拉格朗日中值定理。
Paper math work.

Day 4
Finished chapter auto-gradient. Learnt gradient calculation in torch, warm up with the linear algebra.  
Learnt the Jacobian matrix and Jacobian vector. And how they used in the gradient calculation.  
The gradient cal aim to optimize the loss function. Loss = L(y), y = f(x).
The target is cal gradient of Loss function over input x. Chain rule is utlized here.  
The auto-gradient calculation in torch. The gradient is saved with "require-gradient=True".
The computation graph is saved during the forward pass.  
The gradients are calculated and saved when `backward()` is executed.  
This distinction is fundamental for understanding PyTorch autograd.