# **ChatGPT SUCKS at addition!!!**

A simple regression model doesn't even NEED neurons to do addition! 🤑 

But LLMs operate as tokenized classifiers, so even something like **carrying** means a precise, continuous signal has to model a discrete bit-state - which can **drift** over the layers. 🤬

So **that's** why your trillion dollar almost-AGI Codex 6.7 can barely do grade-school arithmetic. 

But papers like "Grokking" show how transformers can discover 🧠 a geometric logic for arithmetic, which brings us to the **Manifold Hypothesis**: The idea that complex data concentrates around lower-dimensional structures.

🤓☝️LLMs can learn addition by discovering a **helical** manifold, where digits are encoded as angles on a circle, and magnitude as altitude. 

**I wanted to see for myself**...
...so I trained a 3x8 MLP until it was perfectly adding up to 300 - but it failed on [301, 600]. A PCA on different input activations shows us why: 

- ❌ Instead of a manifold we see bumpy, **nonperiodic** ribbons that look like GeLU curves. 
- ❌ The clusters show **some** generalization, but this likely only reflects the symmetry of the data (a + b = b + a). 💀

...in any case, I think I trained up an expensive mod-lookup table. 🤦‍♂️

**So I stepped it up.**
by training a 15M SLM on addition over [**-1e21, 1e21**] until near-0 loss!
No way it's memorizing THAT, right?

- ✅ Detrending and running FFT on the activations showed **MASSIVE** spikes at specific frequencies, which was already lookin' good. 😏
- ✅ A regression exposed **circular clustering** in Fourier space and a linear r-score of 0.87. 

Graphing these projections finally revealed my helix. I'm glad this introductory project to mechanistic interpretability turned out to be a success. 🤑

