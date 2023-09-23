# Fine tuning meta llma2
## Why Fine-tune?
* There are three main reasons why you'd consider fine-tuning a large language model:

* Reduce hallucinations particularly when you pose questions the model hasn't seen in its training data
Make the model suitable for a particular use case, for example, fine-tuning on private company data
* To remove or add undesirable and desirable behavior
 # Large Language Model Fine-tuning Strategies
Several methods have been proposed for fine-tuning large language models. One of them is `LoRA(Low-Rank Adaptation of Large Language Models)`.

_LoRA_ allows you to train weights specific to your use case and later merge them with the original model. The fact that you are training fewer weights compared to all the model weights makes it possible to use LoRA to fine-tune large language models on a single GPU.

_Fine-tuning_ Large Language Models With LoRA
`LoRA` works by freezing the weights of the language model and introducing new matrices into the transformer layers, reducing the number of trainable parameters and making fine-tuning possible with less GPU compute. This is because there is less memory requirement. LoRA is different from prior methods because it doesn't introduce inference latency.

LoRA trains dense layers in the neural network indirectly through rank decomposition matrices of the dense layers. As shown in the following image, LoRA only trains the A and B matrices, leaving the pre-trained weights frozen.


### LoRA
> LoRA makes it possible to use the same model for different tasks by swapping the LoRA weights, reducing the storage required for storing different models. Training with LoRA is also faster because only the LoRA matrices are being optimized, unlike full fine-tuning. The method can also be applied with other methods as you will see during fine-tuning.

The formula for computing the low-rank decomposition is:


where:
> $W_0$ is the pre-trained weight matrix
 $∆W$ is the accumulated gradient update during adaptation
 $r$ is the rank of the LoRA module, a number that you can tune during training
$W_0$ is frozen during training while $A$ and $B$ contain the trainable parameters. $A $is initialized using a random Gaussian while $B$ is set to zero at the beginning of training. For simplicity, LoRA is only applied to the query and value matrices of the transformer, meaning that the multi-layer perceptron is frozen and only the attention weights are adapted.

> In `LoRA` a small set of trainable parameters–adapters– are introduced in the model while the pre-trained weights remain frozen. The loss function is optimized by passing the gradient through the frozen model into the adapters.
` inspired by: derrick from MLnuggets` 
