---
layout: post
title: Averaging M out of N vectors using PyTorch
date: 2025-05-08
categories: data science implementation
---

<img src="/assets/images/averaging_vectors.png" alt="average vectors" width="500" height="auto" style="display: block; margin-left: auto; margin-right: auto; margin-bottom: 2rem;">

In data science, you may end up with $$ N $$ vectors but only need to average $$ M $$ of them. A common scenario is when working with padding, where you may need to ignore vectors corresponding to padded indices. This operation should ideally be __computationally fast__ and __differentiable__ to support backpropagation (see [[1]](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)).

In this blog post, we’ll show a differentiable way to calculate the average of $$ M $$ out of $$ N $$ vectors using PyTorch. While we use PyTorch for all examples, the approach also works with other libraries that support linear algebra, like [NumPy](https://numpy.org/) or [TensorFlow](https://www.tensorflow.org/).

To illustrate the concept, we’ll use 4 vectors of size 3. The method, however, generalizes to any number of vectors and dimensions.

### Averaging

Let’s assume the vectors are arranged as columns in a matrix:

$$
\begin{pmatrix}
\vec{ v_0 } & \vec{ v_1 } & \vec{ v_2 } & \vec{ v_3 }
\end{pmatrix}=
\begin{pmatrix}
v_{00} & v_{10} & v_{20} & v_{30} \\
v_{01} & v_{11} & v_{21} & v_{31} \\
v_{02} & v_{12} & v_{22} & v_{32} \\
\end{pmatrix}
$$

In PyTorch, you can create a simple example of such a matrix via

{% highlight python %}
vectors = torch.arange(0, 12).reshape(3, 4).float()
{% endhighlight %}

To compute the mean of all vectors

$$
\begin{pmatrix}
\frac{1}{4}(v_{00} + v_{10} + v_{20} + v_{30}) \\
\frac{1}{4}(v_{01} + v_{11} + v_{21} + v_{31}) \\
\frac{1}{4}(v_{02} + v_{12} + v_{22} + v_{32})
\end{pmatrix}
$$

you can use

{% highlight python %}
vectors_mean = torch.mean(input=vectors, dim=1, keepdim=True)
{% endhighlight %}

Suppose you only want to average vectors 0, 2, and 3, the expected result would be

$$
\begin{pmatrix}
\frac{1}{3}(v_{00} + v_{20} + v_{30}) \\
\frac{1}{3}(v_{01} + v_{21} + v_{31}) \\
\frac{1}{3}(v_{02} + v_{22} + v_{32})
\end{pmatrix} \tag{1}
$$

To do this, define a mask that indicates which vectors to include. The mask has the same size as the number of vectors (4 in this case), with 1s indicating vectors to include and 0s for those to ignore.

$$
\begin{pmatrix}
1 \\
0 \\
1 \\
1
\end{pmatrix}
$$

This is defined in PyTorch via

{% highlight python %}
mask = torch.tensor([[1], [0], [1], [1]])
{% endhighlight %}

We sum the elements of the mask and divide each element by this sum, which norms the mask

$$
\begin{pmatrix}
\frac{1}{3} \\
0 \\
\frac{1}{3} \\
\frac{1}{3}
\end{pmatrix}
$$

and in PyTorch

{% highlight python %}
mask_sum = torch.sum(mask)
normed_mask = mask / mask_sum
{% endhighlight %}

Now, we perform a matrix-vector multiplication (see [[2]](https://en.wikipedia.org/wiki/Matrix_multiplication))

$$
\begin{pmatrix}
v_{00} & v_{10} & v_{20} & v_{30} \\
v_{01} & v_{11} & v_{21} & v_{31} \\
v_{02} & v_{12} & v_{22} & v_{32} \\
\end{pmatrix} \cdot
\begin{pmatrix}
\frac{1}{3} \\
0 \\
\frac{1}{3} \\
\frac{1}{3}
\end{pmatrix}
$$

which can be computed in PyTorch as

{% highlight python %}
vectors_0_2_3_mean = torch.matmul(vectors, normed_mask)
{% endhighlight %}

We retrieve the desired outcome (1).

To compute the mean over all vectors, like above, we use a mask of all ones:

$$
\begin{pmatrix}
1 \\
1 \\
1 \\
1
\end{pmatrix}
$$

which is normed to

$$
\begin{pmatrix}
\frac{1}{4} \\
\frac{1}{4} \\
\frac{1}{4} \\
\frac{1}{4}
\end{pmatrix}
$$

<code>torch.matmul</code>, <code>torch.sum</code> and element-wise division are all computationally fast and differentiable operations, making them well-suited for use in neural networks with backpropagation.

### How to Create Masks?

To create an alternating mask

$$
\begin{pmatrix}
0 \\
1 \\
0 \\
1
\end{pmatrix}
$$

you can create a vector from index 0 to 3

$$
\begin{pmatrix}
0 \\
1 \\
2 \\
3
\end{pmatrix}
$$

and apply an element-wise modulo operation. Since <code>True</code> is equivalent to 1 and <code>False</code> is equivalent to 0 (when converting from <code>Boolean</code> to <code>Integer</code>), we end up with the alternating mask

{% highlight python %}
alternating_mask = (torch.arange(0, 4).reshape(4, 1) % 2).short()
{% endhighlight %}

For creating a mask that is 1 up to a certain index (here index 2, excluded)

$$
\begin{pmatrix}
1 \\
1 \\
0 \\
0
\end{pmatrix}
$$

we also create a vector from index 0 to 3 and then apply an element-wise comparison against the target index; the result will be 1s for all positions less than the target and 0s otherwise.

{% highlight python %}
mask_up_to_idx = (torch.arange(0, 4).reshape(4, 1) < 2).short()
{% endhighlight %}

### Summary

In this blog post, we discussed an approach to calculate the average of $$ M $$ out of $$ N $$ vectors using masking and matrix multiplication. All operations are computationally fast and differentiable, making them well-suited for backpropagation in neural networks.

### References

* [1] [PyTorch: autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
* [2] [Wikipedia: Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)
