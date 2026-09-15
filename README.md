<div align="center">

🧠 Neural Networks, Understood

From a single neuron to hidden layers, backpropagation & Adam

A visual-first TensorFlow learning note — focused on why each piece exists.

<br>






</div>

⚡ The Entire Idea in 20 Seconds

Raw Input
   │
   ▼
┌───────────────┐
│    Neurons    │  ← weights decide what relationships matter
└───────┬───────┘
        ▼
┌───────────────┐
│ Hidden Layer  │  ← multiple neurons build useful representations
└───────┬───────┘
        ▼
┌───────────────┐
│ ReLU / Other  │  ← adds non-linearity
└───────┬───────┘
        ▼
     Prediction
        │
        ▼
       Loss      ← how wrong were we?
        │
        ▼
 Backpropagation ← calculate gradients
        │
        ▼
       Adam      ← use gradients to update parameters
        │
        └───────────────► repeat 🔁

Mental model: Neurons transform data → hidden layers build representations → loss measures the mistake → backprop finds gradients → Adam improves the weights.

🗺️ Roadmap

Stage

Question we are answering

🧩 Neuron

How does one neuron process input?

🧠 Multiple Neurons

Why isn't one neuron enough?

🕵️ Hidden Layer

What happens between input and output?

⚡ ReLU

Why do hidden layers need non-linearity?

❌ XOR

What can multiple neurons do that one linear rule cannot?

📉 Loss

How does the network know it is wrong?

↩️ Backpropagation

How do we find which weights caused the error?

🧭 Gradient

In which direction should a parameter move?

🚀 Adam

How are those parameters actually updated?

01 — Start With One Neuron

A neuron is basically a learnable mathematical function.

It receives inputs:

x₁ ──┐
x₂ ──┼──► [ NEURON ] ──► output
x₃ ──┘

and calculates:

$$
z = w_1x_1 + w_2x_2 + w_3x_3 + b
$$

or more compactly:

$$
z = Wx+b
$$

Tiny example

Inputs   → x₁ = 2,   x₂ = 3
Weights  → w₁ = 0.5, w₂ = 0.2
Bias     → b = 1

Therefore:

$$
z=(2\times0.5)+(3\times0.2)+1=2.6
$$

So what is actually learned?

x = data              → given by us
w = weights           → learned
b = bias              → learned

The weights control how strongly each input affects this neuron.

02 — Why Do We Need Multiple Neurons?

This was the important question:

If one neuron already has weights and bias, why create 4, 10, 100 neurons?

Because one neuron gives us only one learned transformation of the input.

Imagine three inputs:

Hours Studied ───┐
Attendance ──────┼──► ONE neuron ──► one signal
Previous Score ──┘

Now use four neurons:

                    ┌──► Neuron 1 ──► h₁
                    │
3 Input Features ───┼──► Neuron 2 ──► h₂
                    │
                    ├──► Neuron 3 ──► h₃
                    │
                    └──► Neuron 4 ──► h₄

Every neuron sees the same inputs, but owns different weights:

$$
h_1=f(W_1x+b_1)
$$

$$
h_2=f(W_2x+b_2)
$$

$$
h_3=f(W_3x+b_3)
$$

So the network gets multiple different views / transformations of the same data.

💡 Core Idea

Dense(4) doesn't mean 4 data points.
It means 4 neurons → 4 learned output signals.

03 — What Is a Hidden Layer Really Doing?

keras.layers.Dense(4, activation="relu")

means:

Input
  │
  ▼
┌──────────────────────────────┐
│         HIDDEN LAYER         │
│                              │
│  N₁    N₂    N₃    N₄       │
│  │     │     │     │        │
│  h₁    h₂    h₃    h₄       │
└──────────────┬───────────────┘
               │
               ▼
        Next Layer / Output

The hidden layer's job is to turn:

raw features

into:

more useful internal features / representations

For example:

$$
[x_1,x_2,x_3]
$$

may become:

$$
[h_1,h_2,h_3,h_4]
$$

The next layer no longer has to work directly with only the original input — it can work with the representation created by the hidden neurons.

Important

A neuron does not necessarily learn a clean human-readable concept such as:

Neuron 1 = attendance detector
Neuron 2 = study detector

That is only a useful intuition.

Real neural networks often learn representations distributed across many neurons.

04 — Then What Exactly Does ReLU Do?

The neuron first calculates:

$$
z=Wx+b
$$

Then ReLU calculates:

$$
ReLU(z)=\max(0,z)
$$

So:

Before ReLU

After ReLU

-8

0

-2

0

0

0

3

3

9

9

Visually:

Neuron calculation
       │
       ▼
       z
       │
       ▼
    ┌──────┐
    │ ReLU │
    └──┬───┘
       │
   ┌───┴─────────────┐
   │                 │
z ≤ 0             z > 0
   │                 │
   ▼                 ▼
   0                 z

Weights + bias decide what relationship the neuron responds to.
ReLU changes the neuron's response by clipping negative values to zero and introduces non-linearity.

That second part is crucial.

05 — Why Does Non-Linearity Matter?

Suppose we stack only linear transformations:

$$
h=W_1x+b_1
$$

then:

$$
y=W_2h+b_2
$$

Substitute the first equation:

$$
y=W_2(W_1x+b_1)+b_2
$$

which can again be written as:

$$
y=W'x+b'
$$

😐 Still linear.

So:

Linear
  ↓
Linear
  ↓
Linear

can collapse into effectively:

ONE linear transformation

But:

Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU

cannot generally be collapsed into one linear equation.

That's where much of the expressive power comes from.

06 — XOR: The Cleanest Hidden-Layer Example

Consider:

x₁

x₂

XOR

0

0

0

0

1

1

1

0

1

1

1

0

We want output 1 only when the inputs are different.

A single linear rule cannot separate these four original input points correctly.

Let's build two hidden neurons manually

🟣 Hidden Neuron 1

$$
h_1=ReLU(x_1-x_2)
$$

Weights:

[1, -1]

This activates for:

[1, 0]

🟢 Hidden Neuron 2

$$
h_2=ReLU(x_2-x_1)
$$

Weights:

[-1, 1]

This activates for:

[0, 1]

Watch the representation change

Input

h₁

h₂

h₁ + h₂

[0,0]

0

0

0

[0,1]

0

1

1

[1,0]

1

0

1

[1,1]

0

0

0

🔥 That's XOR.

          x₁,x₂
            │
       ┌────┴────┐
       ▼         ▼
 ReLU(x₁-x₂)  ReLU(x₂-x₁)
       │         │
       h₁        h₂
       └────┬────┘
            ▼
         h₁ + h₂
            │
            ▼
           XOR

This is the important insight:

The hidden neurons transformed a difficult representation into one where the output became easy to compute.

The exact weights learned during real training do not have to match these hand-designed weights.

07 — Neural Network vs Traditional Machine Learning

Neural networks are not automatically better.

Problem

Often a strong starting point

Small tabular dataset

Logistic Regression / Trees / Boosting

Simple linear relationship

Linear Regression

Hand-engineered structured features

Traditional ML can be excellent

Images

Neural Networks / CNNs

Audio

Neural Networks

Language

Neural Networks / Transformers

Complex representation learning

Neural Networks

The real difference

Traditional ML often looks like:

Raw Data
   │
   ▼
Human Feature Engineering
   │
   ▼
ML Model
   │
   ▼
Prediction

Deep learning often pushes more of the representation learning into the model:

Raw Data
   │
   ▼
Hidden Layers
   │
   ▼
Learned Representations
   │
   ▼
Prediction

Neural networks become especially valuable when learning the representation itself is a major part of the problem.

08 — The Model Predicted Something. Now What?

Suppose:

Actual     = 1
Prediction = 0.31

The network needs a number telling it:

How bad was that prediction?

That's the loss.

For regression, one common example is Mean Squared Error:

$$
MSE=\frac{1}{n}\sum(y-\hat y)^2
$$

For binary classification, we commonly use:

loss="binary_crossentropy"

Loss does not fix the model.

It only measures the mistake.

09 — Backpropagation: Who Caused the Mistake?

Now we have a loss.

The network contains many parameters:

w₁
w₂
w₃
...
biases

We need to know:

How did each parameter affect the loss?

Backpropagation computes derivatives such as:

$$
\frac{\partial Loss}{\partial w}
$$

These derivatives are the gradients.

Prediction
    │
    ▼
   Loss
    │
    ▼
Backpropagation
    │
    ▼
Gradients for every parameter

Backpropagation calculates gradients. It does not itself define the optimizer's update rule.

10 — What Does a Gradient Tell Us?

A gradient answers:

If I change this parameter slightly, how will the loss change?

Example:

$$
\frac{\partial Loss}{\partial w}=+5
$$

The loss is increasing in the positive direction around the current point.

A basic gradient-descent update is:

$$
w_{new}=w_{old}-\eta\frac{\partial Loss}{\partial w}
$$

where:

$$
\eta = learning\ rate
$$

So the gradient gives the optimizer the local slope information needed to improve the parameter.

11 — Where Does Adam Enter?

This was another key distinction:

Backpropagation ≠ Adam

They have different jobs.

Backpropagation

"Give me the gradients."

Adam

"Now that I have the gradients,
 how should I update each parameter?"

Complete flow:

Forward Pass
     │
     ▼
 Prediction
     │
     ▼
    Loss
     │
     ▼
Backpropagation
     │
     ▼
  Gradients
     │
     ▼
    Adam
     │
     ▼
Updated Weights & Biases

12 — Why Adam Instead of a Basic Update?

Basic gradient descent looks roughly like:

$$
w_{new}=w_{old}-learning_rate\times gradient
$$

Adam is more sophisticated.

It keeps moving estimates related to:

gradients

squared gradients

This helps it adapt parameter updates during training.

You don't need to memorize Adam's full equation yet.

For now:

Backprop tells Adam what the gradients are. Adam uses that information plus running gradient statistics to decide the parameter updates.

13 — Put Everything Together

flowchart TD
    A[Input Data] --> B[Neurons: Wx + b]
    B --> C[Activation: ReLU]
    C --> D[Hidden Representation]
    D --> E[Output Layer]
    E --> F[Prediction]
    F --> G[Loss]
    G --> H[Backpropagation]
    H --> I[Gradients]
    I --> J[Adam Optimizer]
    J --> K[Update Weights & Biases]
    K --> B

This loop is training.

14 — TensorFlow Version

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Features:
# [hours_studied, attendance, previous_score]

X = np.array([
    [1, 50, 35],
    [2, 60, 40],
    [3, 65, 45],
    [4, 75, 55],
    [5, 80, 60],
    [6, 85, 70],
    [8, 90, 80],
    [2, 90, 80]
], dtype=np.float32)

# 0 = Fail, 1 = Pass
y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=np.float32)

model = keras.Sequential([
    keras.Input(shape=(3,)),

    # 4 neurons create a hidden representation
    keras.layers.Dense(4, activation="relu"),

    # Binary classification output
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=500,
    verbose=0
)

test_data = np.array([
    [6, 85, 70],
    [2, 50, 35],
    [5, 80, 60]
], dtype=np.float32)

predictions = model.predict(test_data)

print(predictions)

15 — Read This TensorFlow Code Like English

keras.Input(shape=(3,))

Every sample contains 3 features.

Dense(4, activation="relu")

Create 4 hidden neurons.
Each neuron gets all 3 features, owns different weights/bias, and passes its result through ReLU.

Dense(1, activation="sigmoid")

Combine the hidden representation into one binary-classification output between 0 and 1.

loss="binary_crossentropy"

Measure how wrong the binary prediction is.

optimizer="adam"

Use the gradients produced during training to update the parameters.

model.fit(...)

Repeat forward pass → loss → backpropagation → optimizer updates across the training data.

🧠 Final Mental Model

Don't memorize isolated definitions.

Remember the reason each component exists:

┌──────────────────────────────────────────────────────┐
│ NEURON                                               │
│ Learn one weighted transformation of its inputs      │
├──────────────────────────────────────────────────────┤
│ MULTIPLE NEURONS                                     │
│ Learn multiple transformations in parallel           │
├──────────────────────────────────────────────────────┤
│ HIDDEN LAYER                                         │
│ Build a useful internal representation               │
├──────────────────────────────────────────────────────┤
│ ReLU                                                 │
│ Add non-linearity; negative activation becomes zero  │
├──────────────────────────────────────────────────────┤
│ OUTPUT LAYER                                         │
│ Turn the representation into a prediction            │
├──────────────────────────────────────────────────────┤
│ LOSS                                                 │
│ Measure the prediction error                         │
├──────────────────────────────────────────────────────┤
│ BACKPROPAGATION                                      │
│ Calculate gradients                                  │
├──────────────────────────────────────────────────────┤
│ GRADIENT                                             │
│ Describe how loss changes with each parameter        │
├──────────────────────────────────────────────────────┤
│ ADAM                                                 │
│ Use gradients to update weights and biases           │
└──────────────────────────────────────────────────────┘

<div align="center">

One sentence worth remembering

Neural networks learn useful representations with neurons, measure mistakes with loss, find gradients with backpropagation, and improve their parameters with an optimizer such as Adam.

<br>

Input → Representation → Prediction → Error → Gradient → Update → Repeat

🚀 Next

Learning Rate → Epochs & Batches → Train/Test Split → Overfitting → Dropout

</div>
