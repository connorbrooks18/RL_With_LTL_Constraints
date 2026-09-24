For my undergraduate thesis at the University of Iowa, I am working with Prof. Shaoping Xiao on extending https://github.com/alperkamil/csrl to the POMDP case

Initial steps:
  1. Tabular Q-learning -> approximate/neural network
  2. MDP -> POMDP (but just use most recent observation)
  3. GRU to handle observation history 
  4. LSTM? Transformer? Something else?
  5. Do a full model-based system with M(s_i, a_i) -> (s_{i+1}, r_{i+1})
