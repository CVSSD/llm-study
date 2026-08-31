day1-day3
Softmax as well softmin:  
    A math approach of min or max with a error.  
    Find max or min need sort argorithm. Need a lot of calculate when data size is large.  
    Softmax/min offer a approcimite approach. Only need a sum and log.  
    The boundart: max < softmax < max + n * lg(n) 
                  min > softmin > min - n * lg(n)
    Softmax and softmin are smooth approximations of max and min.  
    They replace the non-differentiable max/min operation with a differentiable log-sum-exp function.  
    The approximation error is bounded by log(n)/λ, which decreases as the temperature parameter λ increases.

    When utilize softmax in the normalize of y predict. The softmax show its advan tage in differentiable  
    Also need to pay attention to the boundary of exp(xi) exceed the data type.
    The exp(xi - mean(x)) could help. As the constant term will be cacle out.   
    