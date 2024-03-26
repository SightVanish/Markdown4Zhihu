# Cross-Entropy

Cross-entropy loss is widely used as loss function in classification tasks of machine learning. This article will dig into the fundamental of cross-entropy.

In this digital era, everything in computer is stored via bits (0s & 1s). So what is the minimum length of bits we need to transmit a message? According to Claude Shannon, transmitting 1 bit of information is equivalent to reduce the recipient's uncertainty by a factor of 2. **Entropy is used to measure the uncertainty associated with a random variable.**

## Entropy

For example, randomly *pick up a ball from a box with 4 balls -- one white, one red, one yellow, one blue*. So what is the entropy of the result?

A: The possibility of picking a certain ball is  <img src="https://www.zhihu.com/equation?tex=\frac{1}{4}" alt="\frac{1}{4}" class="ee_img tr_noresize" eeimg="1"> . After receiving the message, we would know exactly which ball is picked which means the possibility of picking such a ball is  <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1"> . As each bit of message will reduce the recipient's uncertainty by a factor of 2, we would need 2 bits to eliminate uncertainty.

Next, let's consider a more complex curriculum -- balls are not evenly distributed. *Randomly pick up a ball from a box with 4 balls -- one white and three blue.* What is the entropy of the result?

A: If a white ball is picked: the possibility of picking a white ball is  <img src="https://www.zhihu.com/equation?tex=\frac{1}{4}" alt="\frac{1}{4}" class="ee_img tr_noresize" eeimg="1">  thus the entropy of this result is  <img src="https://www.zhihu.com/equation?tex=\log_24=2" alt="\log_24=2" class="ee_img tr_noresize" eeimg="1">  bits. If a blue ball is picked: the possibility of picking a blue ball is  <img src="https://www.zhihu.com/equation?tex=\frac{3}{4}" alt="\frac{3}{4}" class="ee_img tr_noresize" eeimg="1">  thus the entropy is  <img src="https://www.zhihu.com/equation?tex=\log_2\frac{4}{3}" alt="\log_2\frac{4}{3}" class="ee_img tr_noresize" eeimg="1">  (as 1 bit reduces uncertainty by a factor of 2). The entropy of this event is actually the expected entropy of all possible result:  <img src="https://www.zhihu.com/equation?tex=H=\frac{1}{4}*\log_24+\frac{3}{4}*\log_2\frac{4}{3}=-\frac{1}{4}*\log_2\frac{1}{4}-\frac{3}{4}*\log_2\frac{3}{4}=0.81" alt="H=\frac{1}{4}*\log_24+\frac{3}{4}*\log_2\frac{4}{3}=-\frac{1}{4}*\log_2\frac{1}{4}-\frac{3}{4}*\log_2\frac{3}{4}=0.81" class="ee_img tr_noresize" eeimg="1">  bits, which means you will get about 0.81 bits of information from each draw result.

(Note: "0.81 bits" is theoretically the minimum length of bits to convey message. If you want to sendthe result, you only need one bit, 1 for white ball and 0 for blue ball, but on average the information content is only 0.81 bits.)

To formalize this idea, for a event with  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  possible outcomes and each with a possibility  <img src="https://www.zhihu.com/equation?tex=p_i" alt="p_i" class="ee_img tr_noresize" eeimg="1">  where  <img src="https://www.zhihu.com/equation?tex=i\in \{1,2,\cdots, n\}" alt="i\in \{1,2,\cdots, n\}" class="ee_img tr_noresize" eeimg="1"> , the entropy is defined as:

<img src="https://www.zhihu.com/equation?tex=H(\bold{p}) = -\sum_i^np_i*\log_2(p_i)
" alt="H(\bold{p}) = -\sum_i^np_i*\log_2(p_i)
" class="ee_img tr_noresize" eeimg="1">

## Understanding entropy via encoding.

Suppose there are *3 red balls, 3 blue balls, 2 yellow balls, 1 white balls and 1 black balls.* If we encode the result in one hot: *000 for red balls, 001 for blue balls, 010 for yellow balls, 011 for white balls and 100 for black balls*, the average length of message we sent is 3. However the entropy of this event is: 

<img src="https://www.zhihu.com/equation?tex=H(X) = -\left(\frac{3}{10} \log_2 \frac{3}{10} + \frac{3}{10} \log_2 \frac{3}{10} + \frac{2}{10} \log_2 \frac{2}{10} + \frac{1}{10} \log_2 \frac{1}{10} + \frac{1}{10} \log_2 \frac{1}{10}\right)\approx2.187 \ bits
" alt="H(X) = -\left(\frac{3}{10} \log_2 \frac{3}{10} + \frac{3}{10} \log_2 \frac{3}{10} + \frac{2}{10} \log_2 \frac{2}{10} + \frac{1}{10} \log_2 \frac{1}{10} + \frac{1}{10} \log_2 \frac{1}{10}\right)\approx2.187 \ bits
" class="ee_img tr_noresize" eeimg="1">
So, in general, 3 bits length message will be sent but only 2.187 bits are useful. A better way should be: *0 for red balls, 10 for blue balls, 110 for yellow balls, 1110 for white balls and 1111 for black balls*. The average message length is  <img src="https://www.zhihu.com/equation?tex=0.3 \times 1 + 0.3 \times 2 + 0.2 \times 3 + 0.1 \times 4 + 0.1 \times 4 =2.3 \ bits" alt="0.3 \times 1 + 0.3 \times 2 + 0.2 \times 3 + 0.1 \times 4 + 0.1 \times 4 =2.3 \ bits" class="ee_img tr_noresize" eeimg="1">  which is much more closer to entropy (theoretical optimal length).

Note this kind of encoding method is unambiguous, which means there is only one way to interpret a sequence of bits. (To ensure that decoding is unambiguous, you need to design a prefix-free code, also known as a Huffman code. A prefix-free code is a uniquely decodable code in which no code word is a prefix of another. )

## Cross-Entropy

In reality world, unlike the balls in a box, we do not know exactly the possibility of a certain event. If we want to convey message efficiently, we can utilize the idea of entropy based on the expected possibility. For a result with predicted possibility  <img src="https://www.zhihu.com/equation?tex=q_i" alt="q_i" class="ee_img tr_noresize" eeimg="1"> , a proper encoded message length would be the entropy of this result  <img src="https://www.zhihu.com/equation?tex=-log_2(q_i)" alt="-log_2(q_i)" class="ee_img tr_noresize" eeimg="1"> . Cross-entropy is the average message length we sent:

<img src="https://www.zhihu.com/equation?tex=H(\bold p, \bold q) = -\sum_i^n p_i*\log_2(q_i)
" alt="H(\bold p, \bold q) = -\sum_i^n p_i*\log_2(q_i)
" class="ee_img tr_noresize" eeimg="1">
where  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  is the real distribution and   <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  is the predicted distribution. If predicted distribution is close to real distribution,  <img src="https://www.zhihu.com/equation?tex=H(\bold p, \bold q)" alt="H(\bold p, \bold q)" class="ee_img tr_noresize" eeimg="1">  will be close to real entropy  <img src="https://www.zhihu.com/equation?tex=H(\bold p)" alt="H(\bold p)" class="ee_img tr_noresize" eeimg="1"> .

## Kullback-Leibler Divergence

KL divergence is the gap between cross-entropy and the entropy, noted as:

<img src="https://www.zhihu.com/equation?tex=D_{KL}(\bold p ||\bold q) = H(\bold p, \bold q) - H(\bold p)
" alt="D_{KL}(\bold p ||\bold q) = H(\bold p, \bold q) - H(\bold p)
" class="ee_img tr_noresize" eeimg="1">
Cross-entropy is widely used in machine learning, especially in classification tasks, as cost functions. 

## Why Cross-Entropy

The cross-entropy loss is derived from negative log-likelihood (MLE) of a Bernoulli distribution.

Bernoulli distribution:

In binary classification, we model the probability of a success as  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  and the possibility of observing a failure as  <img src="https://www.zhihu.com/equation?tex=1-p" alt="1-p" class="ee_img tr_noresize" eeimg="1"> . Thus the probability mass function (PMF, possibility of a certain outcome) is:

<img src="https://www.zhihu.com/equation?tex=P(Y=y) = p^y(1-p)^{1-y}
" alt="P(Y=y) = p^y(1-p)^{1-y}
" class="ee_img tr_noresize" eeimg="1">
where  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  is the label (0 or 1).

Negative likelihood of Bernoulli distribution is:

<img src="https://www.zhihu.com/equation?tex=-\log P(Y=y) = -\log(p^y(1-p)^{1-y}) = -(y*\log(p)+(1-y)*\log(1-p))
" alt="-\log P(Y=y) = -\log(p^y(1-p)^{1-y}) = -(y*\log(p)+(1-y)*\log(1-p))
" class="ee_img tr_noresize" eeimg="1">
Viewing  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  as real distribution and  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  as predicted distribution, negative likelihood of Bernoulli distribution is just the same with cross-entropy. (The multi-class cross-entropy loss is a generalization of binary cross-entropy.)

From the very beginning, a widely used loss function is MSE (mean squared error), defined as

<img src="https://www.zhihu.com/equation?tex=L = \frac{||y-\hat y||^2}{2}
" alt="L = \frac{||y-\hat y||^2}{2}
" class="ee_img tr_noresize" eeimg="1">
where  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  is the ground truth and  <img src="https://www.zhihu.com/equation?tex=\hat y" alt="\hat y" class="ee_img tr_noresize" eeimg="1">  is the predicted label.

## Why You Should Use MSE in Classification Tasks

The mean squared error (MSE) loss is related to the maximum likelihood estimation (MLE) under the assumption of norm distribution, which is widely used in regression tasks.

Suppose the error  <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1">  (difference between the ground truth and predicted label  <img src="https://www.zhihu.com/equation?tex=\epsilon = y - \hat y" alt="\epsilon = y - \hat y" class="ee_img tr_noresize" eeimg="1">  ) follows norm distribution with mean  <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1">  and a constant variance  <img src="https://www.zhihu.com/equation?tex=\sigma^2" alt="\sigma^2" class="ee_img tr_noresize" eeimg="1"> :  <img src="https://www.zhihu.com/equation?tex=\epsilon \sim N(0, \sigma^2)" alt="\epsilon \sim N(0, \sigma^2)" class="ee_img tr_noresize" eeimg="1"> .

The probability density function (PDF) of  <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1">  is:

<img src="https://www.zhihu.com/equation?tex=f(\epsilon) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{\epsilon^2}{2\sigma^2}}
" alt="f(\epsilon) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{\epsilon^2}{2\sigma^2}}
" class="ee_img tr_noresize" eeimg="1">
The likelihood function is:

<img src="https://www.zhihu.com/equation?tex=L(\theta)=f(\epsilon; \theta)=f(y-\hat y;\theta)
" alt="L(\theta)=f(\epsilon; \theta)=f(y-\hat y;\theta)
" class="ee_img tr_noresize" eeimg="1">
where  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  represents the parameters of the model.

MLE aims to find the parameter values that maximize the likelihood function.

<img src="https://www.zhihu.com/equation?tex=\begin{align*}
  \arg\max\limits_\theta L(\epsilon; \theta)
    &= \arg\max\limits_\theta \log L(\epsilon; \theta) \\
    &= \arg\max\limits_\theta \log L(y - \hat y; \theta) \\
    &= \arg\max\limits_\theta (-\frac{1}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}(y-\hat y)^2) \\
    &= \arg\min\limits_\theta \frac{1}{2}(y-\hat y)^2
\end{align*}
" alt="\begin{align*}
  \arg\max\limits_\theta L(\epsilon; \theta)
    &= \arg\max\limits_\theta \log L(\epsilon; \theta) \\
    &= \arg\max\limits_\theta \log L(y - \hat y; \theta) \\
    &= \arg\max\limits_\theta (-\frac{1}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}(y-\hat y)^2) \\
    &= \arg\min\limits_\theta \frac{1}{2}(y-\hat y)^2
\end{align*}
" class="ee_img tr_noresize" eeimg="1">
where  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  is implicitly presented within  <img src="https://www.zhihu.com/equation?tex=\hat y" alt="\hat y" class="ee_img tr_noresize" eeimg="1">  and  <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1">  is a constant.

MLE is designed based on norm distribution and is not suitable for classification tasks. Several issues will prevent you from considering MLE as loss function for classification:

- Probabilistic Interpretation: MSE does not inherently consider these the outputs of model as probabilities.
- Non-Convex: the optimization landscape of MSE loss in classification tasks (with sigmoid activation node) is not convex.
- Vanishing Gradient: for a well trained model, the gradient of MSE loss may approach to zero.