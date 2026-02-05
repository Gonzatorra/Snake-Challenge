# Snake RL Agent

This project was developed in collaboration with Nora Ibarguren and focuses on training a Reinforcement Learning (RL) agent capable of successfully playing the classic Snake game.

The main objective is to design, implement, and evaluate different RL algorithms that allow the agent to learn optimal strategies for maximizing its score while avoiding collisions. The environment provides continuous feedback to the agent based on its actions, which encourages exploration, efficient path planning, and adaptive decision-making.

# Approach and Experiments

Several approaches were tested to train the RL agent and improve its performance in the Snake environment:

- ***Simple MLP (input → 256 → 128 → action)***: Behavior Cloning (BC) achieved very low loss, but the agent showed no actual learning or exploration.

- ***BC + PPO with Fine-Tuning***: Using the BC model as a starting point and applying Proximal Policy Optimization (PPO) with various hyperparameter configurations led to better results in terms of exploration and training stability.

- ***LSTM Policy (Recurrent PPO)***: Introduced a recurrent policy to enhance the agent’s memory and decision consistency over time. However, this approach was more computationally demanding.

- ***Reward Shaping and Normalization***: Adjusting reward functions and normalizing inputs improved loss behavior and produced more stable learning.

- ***Environment Modifications***: Alterations in the environment setup caused performance degradation, suggesting sensitivity to environmental conditions.

# Additional Resources

For more information and a visual overview of our work, you can view the project presentation here: [Project Presentation](https://docs.google.com/presentation/d/1Il2Ye7n9zEMAomaRvZLAOjK2HHVSXH6bwAYZBaDmuhw/edit?usp=sharing)
