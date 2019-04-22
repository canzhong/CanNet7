# Matrix Capsules with EM Routing using ResNeXt feature sampling
A Pytorch implementation variation of [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) using 4 ResNeXt block as the feature map extractor instead of using the tradition CNN used in the base architecture.

Architecture :

ResNextB(in_channels=1, out_channels=F, stride=1, cardinality=4, base_width=16, widen_factor=2)

ResNextB(in_channels=F, out_channels=G, stride=1, cardinality=4, base_width=16, widen_factor=2)

ResNextB(in_channels=G, out_channels=H, stride=1, cardinality=4, base_width=16, widen_factor=2)

ResNextB(in_channels=H, out_channels=A, stride=1, cardinality=4, base_width=16, widen_factor=2)

PrimaryCaps(A, B, 1, P, stride=1)

ConvCaps(B, C, K, P, stride=2, iters=iters)

ConvCaps(C, D, K, P, stride=1, iters=iters)

ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)

F=G=H=A=B=C=D=16, E=10

## MNIST10 experiments

The experiments are conducted on Nvidia Tesla V100.
Specific setting is `lr=0.075`, `batch_size=16`, `weight_decay=0`, Adam optimizer

Following is the result after 40 epochs training:

| Arch | Iters | Coord Add | Loss | BN | Test Accuracy |
| ---- |:-----:|:---------:|:----:|:--:|:-------------:|
| A=B=C=D=16 | 3 | Y | Spread    | Y | 98.81 |

The training time of `A=B=C=D=16` for a 16 batch is around `0.0260s`.

Hyperparameter lr was accidently set to 0.075. I had intended to use a lower learning rate 0.0075. However, it appears that the high lr (compared to the rest of my experiments which had 0.01) did not affect the negative outcome of the test results. Whether or not it had a positive impact on the accuracy could not determine because a second run with the correct lr was not ran because of financial reasons.

## Reference
The research done is solely for non-profit academic purposes. Code from several repos were altered and pieced together to create an architecture compatible with the experiment hypothesis.

Capsule components were inspired from https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch
ResNeXt components were inspired from https://github.com/prlz77/ResNeXt.pytorch
