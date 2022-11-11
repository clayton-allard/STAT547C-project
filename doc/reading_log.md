## Ivan Kobyzev, Simon J.D. Prince, and Marcus A. Brubaker. (2020 June 6). *Normalizing Flows: An Introduction and Review of Current Methods*. 
# November 9, 2022

- A *generative model* is made from data points coming from an unknown probability distribution. *Normalizing flows* is a form of generative modelling.
- Need to be familiar with what a latent variable is. Comes from expectation maximization.
  * **GAN** Generative Adverserial Network
  * **VAE** Variational Auto-Encoder.

I had a ton of notes here that didn't save :(

Recall that is we have a random variable $Z \in \mathbb{R}^d$ and let $Y=g(Z)$ where $g$ is an invertible, differentiable function, then $$p_Y(y)=p_Z(F(y)) |\det Df(y)|=p_Z(F(y)) |\det Dg(f(y))|^{-1}.$$

## Ivan Kobyzev, Simon J.D. Prince, and Marcus A. Brubaker. (2020 June 6). *Normalizing Flows: An Introduction and Review of Current Methods*. 
# November 10, 2022

- We have measurable spaces $(\mu,\mathcal{Z},\Sigma_\mathcal{Z})$ and $(\nu,\mathcal{Y},\Sigma_\mathcal{Y})$ where $\nu=g_\*\mu(U)=\mu(g^{-1}(U))$ for all $U \in \Sigma_\mathcal{Y}$. $\nu$ is known as the pushforward measure. Here we have that $g : \mathcal{Z} \rightarrow \mathcal{Y}$ is a function which acts as a generator and $\mathcal{Z}$ is a latent space.
  * We will assume that all sigma algebras are borel and all measures are absolutely continuous with respect to the lebesgue measure.
- A function $g : \mathcal{Z} \rightarrow \mathcal{Y}$ is a **diffeomorphism** if it is bijective, differentiable, and it inverse is differentiable as well.
- For normalizing flows, we need $g$ to be a diffeomorphism. Although, it only needs to be differentiable almost everywhere to allow for piecewise functions to work.
- We have a parameter space $\Theta=(\theta,\phi)$ where $\theta$ is the paramter vector for $g$ and $\phi$ is the parameter vector for $p_Z$. We also have the sapmle space $\mathcal{D}=\{y^{(i)}\}\_{i=1}^M$. The the goal is to maximize the likelihood $p(\mathcal{D}\~| \Theta)$ or the log likelihood $$p(\mathcal{D}~| \Theta)=\sum_{i=1}^M\log p_Y(y^{(i)}\~| \Theta).$$
- Suppose we have a latent variable model $p(x)=\int p(x,y)dy$ where $x$ is an observed variable and $y$ is the latent variable. The posterior distribution $p(y\~|x)$ is used when estimating the parameters of the model, but it is intractable in practice usually. Another approach is to do variational inference using $q(y\~|x,\theta)$. To make this as close to $p$ as possible, we minimize the KL divergence. One can reparametrize $q(y\~|x,\theta)=p_Y(y\~|\theta)$ with normalizing flows.
