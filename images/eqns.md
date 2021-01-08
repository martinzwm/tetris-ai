DQN objective:

$$\large J(\textbf{w}) = \textbf{E}_{(s_t, a_t, r_t, s_{t+1})}[(y_t^{DQN} - \hat{q}(s_t, a_t, \textbf{w}))^2]$$

DQN y:

$$\large y_t^{DQN} = r_t + \gamma \space \underset{a'}{max}[\hat{q}(s_{t+1}, a', \textbf{w}^-]$$

DDQN y:

$$\large y_t^{DoubleQ} = r_t + \gamma \space \hat{q}(s_{t+1}, \underset{a'}{argmax}[\hat{q}(s_{t+1}, a', \textbf{w})], \textbf{w}')$$

PER:

$$\large P(i) = \frac{p_i^a}{\Sigma_k \space p_k^a}$$