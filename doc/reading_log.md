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

## Ivan Kobyzev, Simon J.D. Prince, and Marcus A. Brubaker. (2020 June 6). *Normalizing Flows: An Introduction and Review of Current Methods*. 
# November 11, 2022

- Normalizing flows must satisfy 3 conditions for practicality.
  * Must be invertible. We need $g$ for sampling and $f$ for computing likelihood.
  * Must be sufficiently expressive to model the distribution of interest.
  * Must be computationally efficient.

- A basic bijective non-linear function is $g(x)=(h(x_1),h(x_2),...,h(x_D))$ where $h:\mathbb{R}\rightarrow\mathbb{R}$ is a scalar valued bijection. The inverse of $g$ only requires computing $h^{-1}$ and the Jacobian is the product of the absolute values of the derivatives of $h$. This can be generalized so that each element has its own $h$ function. These are *activation functions*.

- The problem is that we cannot do elementwise transormations because the there is no way to express correlation between dimensions. I don't fully understand this.

- For a linear flow we have $g(x)=Ax+b$ where $A\in \mathbb{R}^{D \times D}.
- The problem with linear flows is that it becomes expensive. It takes $\mathcal{O}(D^3)$ to compute the determinant of a $D \times D$ matrix.
- We want to make restrictions on $A$ so that this can be a more practical process (we will use coupling flows).
- One example is that we can have $A$ be a diagonal matrix. Then it can be computed in linear time.
- Other methods that were tried are using triangular matrices and orthoganal matrices. 

- Coupling and Autoregressive flows are considered to be very important.
- Suppose we take a disjoint partition of the the input $x \in \mathbb{R}^D$ and put them into two subspaces $(x^A,x^B) \in \mathbb{d}\times\mathbb{D-d}$ and a bijective function $h(\cdot;\theta):\mathbb{R}^d\rightarrow\mathbb{R}^d$. Then $g:\mathbb{R}^D\rightarrow\mathbb{R}^D$ can be defined in the following way. We have $$y^A=h(x^A;\Theta(x^B))$$ $$y^B=x^B$$ where the parameter $\theta$ can be defined by any arbitary function $\Theta(x^B))$. The bijection $h$ is a _coupling function_ and the resulting function $g$ is a _coupling flow_.
  * A _coupling flow_ is invertible $\Leftrightarrow h$ is invertible and has inverse $$x^A=h^{-`}(y^A;\Theta(x^B))$$ $$x^B=y^B.$$
  * The advantage of the coupling flows is that we have lots of flexibility with what $\Theta$ can be.
  * In practice, $\Theta$ is usually modelled as a neural network.

- An _autoregressive flow_ can be formed taking a a bijective function $h(\cdot;\theta):\mathbb{R}\rightarrow\mathbb{R}$ and having $g(x)=y$ such that $$y_t=h(x_t;\Theta_t(x_{t:t-1}))$$. Essentially, each subsequent element of $y$ is conditional on the previous elements.
  * By conditioning in this way, we get a triangular matrix which means that the determinant is just the product $$\det(Dg)=\prod_{t=1}^D \frac{\partial y_t}{\partial x_t}.$$

- The main idea with everything is to find a balance between something where the determinant is easy to compute and the inverse is easy to compute.

- Use an _Inverse autoregressive flow (IAF)_ for fast sampling and a _Masked autoregressive flow (MAF)_ for fast density estimation.

- **Universality property** says that the flow can learn any target density to any required precision given sufficient capacity and data.
- **NOTE the proof here. Page 7. Requires measure thoery.** Use triangular matrices and the dominated convergence theorem.

- A scalar coupling functions must be strictly monotone.

## Ivan Kobyzev, Simon J.D. Prince, and Marcus A. Brubaker. (2020 June 6). *Normalizing Flows: An Introduction and Review of Current Methods*. 
# November 13, 2022

- Residual flows have the form $$g(x)=x+F(x)$$ where $F(x)$ is a feedforward network. The main motivation is to save memory during trainingand stabilize computation.
- Similar to the coupling flow, we break up the space into partitions and have $$y^A=x^A+F(x^B)$$ $$y^B=x^B+G(x^A)$$ where $F:\mathbb{R}^{D-d}\rightarrow\mathbb{R}^d$ and $G:\mathbb{R}^d\rightarrow\mathbb{R}^{D-d}$ are residual blocks. This is invertible, but computation of the Jacobian is inefficient.
- Proposition 7 says that a residual connection is invertible if the Lipschitz constant of the residual block has lipschitz constant less than 1.

- The last method to look at is the Infinitesimal (Continuous) Flows. For this, we essentially have so many composed functions that it can be viewed in the continuous scope. We have $$\frac{d}{dt}x(t)=F(x(t),\theta(t))$$ where $F:\mathbb{R}^D \times \Theta \rightarrow\mathbb{R}^D.$
- We can look at this from the scope of ordinary differential equations, and from stochastic differential equations. 

- **ODE methods**
- Take $t \in [0,1]$ with $x(0)=z$ and $x(1)=g(z)=y$. We'll denote $\Phi^t(z)$ for each $t$. 
- At each time $t, \Phi^t(\cdot):\mathbb{R}^D \rightarrow\mathbb{R}^D$ is a diffeomorphism and satisfies the group law $\Phi^t \circ \Phi^s=\Phi^(t+s).$
- An ODE in this context defines a one-parameter group of diffeomorphisms on $\mathbb{R}^D$. Such a group is called a smooth flow.
- When $t=1$, the diffeomorphism $\Phi^1(\cdot)$ is a _time one map_. This goes under the name **Neural ODE (NODE)**. In other words, this is an infinitely deep neural network with $z$ as an input and $y$ as an output and with continuous weights $\theta(t)$. 
- _Adjoint sensitivity method_ is the continuous analog of backpropagation.
  * For loss $L(x(t))$ with $x(t)$ as the solution to the ODE above, we have that the sensitivity is $a(t)=\frac{dL}{dx(t)}$. We define $$a(t)=\frac{dL}{dx(t)}.$$ The back propagation formula then becomes $$\frac{da(t)}{dt}=-a(t)\frac{dF(x(t),\theta(t)))}{dx(t)}.$$
- For density estimation, we do not have a loss function, but we have a likelihood function to maximize. $$\frac{d}{dt}\log(p(x(t)))=-Tr(\frac{dF(x(t))}{dx(t)}).$$
- One problem with using the ODE method is that it must be orientation preserving. This means that the Jacobian **must** be positive.
- The **Augmented Neural ODE** bypasses this by adding in extra variables $\hat{x}(t)\in\mathbb{R}^p$. Then we solve the new ODE $$\frac{d}{dt}(x(t), \hat{x}(t),\theta(t))$$ with initial conditions $x(0)=z$ and $\hat{x}(0)=0$. This gives us the flexibility to let $\hat{x}(t)$ to be some mapping that allows the Jacobian to remain positive.

**Stochastic Differential Equations (SDE) based methods**
- We use the Ito process which describes the change of a random variable $x \in \mathbb{R}^D$ as a function of time $t$. This is $$dx(t)=b(x(t),t)+\sigma(x(t),t)dB\_t$$ where $b(x,t)\in \mathbb{R}^D$ is the drift coefficient, $\sigma(x,t)\in \mathbb{R}^{D\times D} is the diffusion coefficient.
- We can use MCMC methods with $x^t$ being the random variable at time $t$ with $x^0=y$ as the data point and $x^T=z$ as the base point. Then we consider the transition $q(x^t \~| x^{t-1})$ for the forward transition probability and $p(x^{t-1} \~| x^{t})$ for the backward transition. These are thought to either be binomial or normal. 
  * Applying the backward transition, we obtain a new density $p(x^0)$ which we hope to match with $q(x^0)$. 

- Flow++ is the best performing approach for image datasets.
