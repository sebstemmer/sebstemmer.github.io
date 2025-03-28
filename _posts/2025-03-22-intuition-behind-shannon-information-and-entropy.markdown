---
layout: post
title: "Intuition Behind Shannon Information and Entropy"
date: 2025-03-22
categories: jekyll update
---

<img src="/assets/images/throwing_coins.png" alt="Throwing coins." width="500" height="300" style="display: block; margin-left: auto; margin-right: auto; margin-bottom: 2rem;">

In this blog post I will provide you with intuition for the terms information and entropy from [Shannon's](https://de.wikipedia.org/wiki/Claude_Shannon) information theory.

Let's consider an i.i.d. random variable, such as the outcome of flipping two (not necessarily fair) coins

$$ X \in \{00, 01, 10, 11\} $$

The key question to ask yourself is 

How easy is it to predict the realization $$ X = x $$?

For example, suppose we have the following probability distribution

$$ P(X = 00) = 0.4 , P(X = 01) = 0.3,$$
$$ P(X = 10) = 0.2, P(X = 11) = 0.1 $$

and the corresponding question is: How easy is it to predict $$ X = 00 $$?

To answer this, imagine playing the following game: In each round, $$ X $$ is sampled (only I know the outcome) and you must decide whether to say "Not Playing" or "Now $$ X = 00 $$". We play for several rounds and track how often you are correct in the cases where you choose to play.

$$ X $$ is i.i.d. (each realization is independent), which means you cannot apply an intelligent strategy to predict $$ X = 00 $$ and you can just say "Now $$ X = 00 $$" in every round.

On average, your prediction will be correct in $$ 40 \% $$ of the rounds you play. Similarly, if you are supposed to predict $$ X = 10 $$, your prediction will be correct in $$ 20 \% $$ of the rounds you play on average. 

The ability to make accurate predictions depends on the probability of a given realization. The higher the probability, the better your ability to predict.

Now let's play the same game, but this time, I'll tell you the observed value of $$ X $$ before you make your prediction. You now have more __information__ (you know what $$ X $$ actually is), which increases your abililty to predict. If $$ X \neq 00 $$, you choose "Not Playing". When $$ X = 00 $$, you can say "Now $$ X = 00 $$" and be correct every round you play. By telling you the observed value of $$ X $$, I've transferred __information__ to you, improving your ability to predict $$ X = 00 $$.

The core intuition here is: If your ability to predict an observed value of a random variable improves because of something I told you, then __information__ has been transferred from me to you. As a result, your __information__ about the random variable has increased, because your predictive ability has improved. 

The information gained when transitioning from complete uncertainty about the realization of a random variable to knowing its observed value is called __self-information__ in Shannon's notation.

Therefore, the self-information of $$ X = 10 $$ is higher than that of $$ X = 00 $$, because the prediction accuracy increased from $$ 20 \% $$ to $$ 100 \% $$ for $$ X = 10 $$ and from $$ 40 \% $$ to $$ 100 \% $$ for $$ X = 00 $$ after receiving the self-information. In general, the lower the probability of a certain realization, the greater the increase in prediction accuracy after receiving self-information about which realization occurs, and thus, the higher the transferred self-information.

In information theory the term self-information is also called __surprisal__ because the lower the probability of a realization of a random variable, the more surprised you are when you predict it correctly in our game. However, I find the topic more intuitive when approached through the concept of information. Therefore, this blog post focuses on that perspective.

Let's denote the self-information of $$ X = x $$ by $$ I(x) $$ and $$ p_x = P(X = x) $$. From what we've deduced, the transferred information depends on the probability $$ p_x $$:

$$ I(x) = I(p_x) $$

Additionally, $$ I(x) > I(y) \text{ if } p_x < p_y \tag{1} $$.

__Note:__ In the edge case where $$ X $$ is always $$ 00 $$, i.e.,

$$ P(X = 00) = 1,$$
$$ P(X = 01) = P(X = 10) =$$
$$ P(X = 11) = 0 $$

you already know the outcome in each round. Telling you the outcome in each round would not increase your prediction ability. Hence, no information is transferred in telling you $$ x $$ and $$ I(x = 00) = 0 $$ or

$$ I(p_x) = 0 \text{ if } p_x = 1 \tag{2} $$

Let's deepen the intuition by assuming two fair coins. The first coin is described by the random variable $$ X_1 $$, and the second coin by the random variable $$ X_2 $$, i.e.,

$$ P(X = 00) = P(X = 01) = $$
$$ P(X = 10) = P(X = 11) = \frac{1}{4} $$

and

$$ P(X_1 = 0) = P(X_1 = 1) = $$
$$ P(X_2 = 0) = P(X_2 = 1) = \frac{1}{2} $$

Note that $$ X_1 $$ and $$ X_2 $$ are independent.

We play the same game as before, i.e. you have to either choose "Not Playing" or "Now $$ X = 00 $$". However, before you make your prediction, I tell you the observed value of $$ X_1 $$.

If $$ X_1 = 1 $$, you choose "Not Playing", otherwise, you choose "Now $$ X = 00 $$". By transferring the self-information $$ I(x_1) $$ of $$ X_1 $$, your ability to predict the outcome improves from $$ 25 \% $$ (on average) to $$ 50 \% $$ (on average).

If, in addition to the observed value of $$ X_1 $$, I also tell you the realization of $$ X_2 $$, even more information is transferred, because now you know the realization of $$ X $$, and you will predict correctly in all played rounds. This is equivalent to telling you the observed value of $$ X $$.

Therefore, the total information transferred is $$ I(p_x) = I(p_{x_1}) + I(p_{x_2}) $$. Note that $$ p_x = p_{x_1} \cdot p_{x_2} $$ because $$ X_1 $$ and $$ X_2 $$ are independent. Hence:

$$ I(p_{x_1} \cdot p_{x_2}) = I(p_{x_1}) + I(p_{x_2}) \tag{3} $$

It is easy to show that 

$$ I(p) = - log_b(p) \tag{4} $$

satisfies all conditions (1), (2), and (3). In fact, one can even show, that equation (4) is the only continuous, non-trivial solution to (3).

For $$ b = 2 $$ the unit of $$ I(p) $$ is called bits. In our example with two fair coins the self-information of one realization is

$$ I(p = \frac{1}{4}) = - log_2(\frac{1}{4}) = 2 \text{ bit} $$

With $$ b = 2 $$ and the intuition that information is transferred when predicitve ability improves, $$ 1 $$ bit of information means that the number of possible outcomes of a realization has been halfed. This aligns with our previous example, where knowing the realization of $$ X_1 $$ reduced the possible outcomes by half.

The self-information of a realization is specific to that realization. For a probability distribution of a random variable $$ X $$, the next step is to calculate the expected self-information:

$$ H(X) = E[I(X)] = \sum_{x \in X}{p_x \cdot I(p_x)} $$

In Shannon's information theory, $$ H(X) $$ is referred to as __entropy__. Based on our developed intuition, entropy describes the predictability of a random variable $$ X $$. We can also say that entropy measures how uncertain the outcome of a random variable is on average.

Sources:

* [Wikipedia: Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
* [Youtube: A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)