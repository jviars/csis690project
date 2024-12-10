# Created for the completion of an MSc by jviars

import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from memory_profiler import memory_usage
import time

# This is the last iteration that seems to improve accuracy over the baseline SNN a bit.
# The accuracy measurements don't seem to be reliable on RISC-based systems, as such, please use a CISC-based system for the best results.

# Fetch the MNIST dataset (handwritten digits)
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"]

# Preprocessing: Standardize pixel values and split into training and testing sets
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding for NN and maintain compatibility with SNN
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Traditional Neural Network (NN) Implementation
nn_start = time.time()
nn = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, activation='relu', solver='adam', learning_rate_init=0.001, random_state=42)
nn.fit(X_train, y_train)
nn_end = time.time()

# NN prediction and accuracy
y_pred_nn = nn.predict(X_test)
nn_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_nn, axis=1))
nn_time = nn_end - nn_start

# Calculate memory usage for NN
nn_start_mem = memory_usage()[0]
nn.predict(X_test)  # Simulate prediction to measure memory
nn_end_mem = memory_usage()[0]
nn_mem_usage = nn_end_mem - nn_start_mem

# Simulate spike activity metric for NN (as neuron activations)
nn_activations = np.mean(np.abs(nn.predict_proba(X_test)), axis=0)  # Average activation magnitude
nn_spike_like_metric = np.sum(nn_activations)

print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Traditional NN Training Time: {nn_time:.4f} seconds")
print(f"Traditional NN Memory Usage: {nn_mem_usage:.4f} MB")
print(f"Traditional NN Simulated 'Spike Activity' Metric: {nn_spike_like_metric:.4f}")

# Baseline SNN Implementation
n_neurons = 100
baseline_start = time.time()

# Define spiking neuron model for baseline
eqs_baseline = '''
dv/dt = (I-v)/tau : 1 (unless refractory)  # Membrane potential dynamics
I : 1  # Input current
tau : second  # Membrane time constant
'''

neurons_baseline = b2.NeuronGroup(
    n_neurons, eqs_baseline,
    threshold='v>1', reset='v=0', refractory=5*b2.ms
)
neurons_baseline.v = 'rand()'
neurons_baseline.I = 1.5
neurons_baseline.tau = 10 * b2.ms

# Monitor spikes for baseline
spike_mon_baseline = b2.SpikeMonitor(neurons_baseline)

# Synaptic connections for baseline
synapses_baseline = b2.Synapses(
    neurons_baseline, neurons_baseline,
    model='w : 1',
    on_pre='v_post += w',
    on_post='w -= 0.01'
)
synapses_baseline.connect(condition='i != j')
synapses_baseline.w = 0.1

# Run baseline network
network_baseline = b2.Network(neurons_baseline, spike_mon_baseline, synapses_baseline)
network_baseline.run(500 * b2.ms)

baseline_end = time.time()
spike_counts_baseline = np.bincount(spike_mon_baseline.i)
predicted_label_baseline = np.argmax(spike_counts_baseline)
snn_predictions_baseline = [predicted_label_baseline for _ in range(len(y_test))]
snn_accuracy_baseline = accuracy_score(np.argmax(y_test, axis=1), snn_predictions_baseline)
baseline_time = baseline_end - baseline_start

print(f"Baseline SNN Accuracy: {snn_accuracy_baseline:.4f}")
print(f"Baseline SNN Simulation Time: {baseline_time:.4f} seconds")

# Improved SNN Implementation
improved_start = time.time()

# Define improved neuron equations with rate coding and adaptive threshold
eqs_improved = '''
dv/dt = (I-v)/tau : 1 (unless refractory)
dthreshold/dt = (v0 - threshold)/tau_t : 1  # Renamed from thresh to threshold
I = input_value * (1 + noise) : 1
input_value : 1  # Input current from MNIST data
noise : 1  # Random noise term
v0 : 1  # Baseline threshold
tau : second
tau_t : second  # Threshold adaptation time constant
'''

# Network architecture
n_input_neurons = 784  # One neuron per MNIST pixel
n_hidden_neurons = 400  # Increased hidden layer size
n_output_neurons = 10

# Input layer with adaptive threshold
neurons_input = b2.NeuronGroup(
    n_input_neurons, eqs_improved,
    threshold='v>threshold', reset='v=0; threshold=clip(threshold+0.05, 0.8, 1.2)', refractory=2*b2.ms
)
neurons_input.v = 'rand()'
neurons_input.threshold = 1
neurons_input.v0 = 1
neurons_input.tau = 5 * b2.ms
neurons_input.tau_t = 20 * b2.ms

# Hidden layer with adaptive threshold
neurons_hidden = b2.NeuronGroup(
    n_hidden_neurons, eqs_improved,
    threshold='v>threshold', reset='v=0; threshold=clip(threshold+0.05, 0.8, 1.2)', refractory=2*b2.ms
)
neurons_hidden.v = 'rand()'
neurons_hidden.threshold = 1
neurons_hidden.v0 = 1
neurons_hidden.tau = 5 * b2.ms
neurons_hidden.tau_t = 20 * b2.ms

# Output layer with adaptive threshold
neurons_output = b2.NeuronGroup(
    n_output_neurons, eqs_improved,
    threshold='v>threshold', reset='v=0; threshold=clip(threshold+0.05, 0.8, 1.2)', refractory=2*b2.ms
)
neurons_output.v = 'rand()'
neurons_output.threshold = 1
neurons_output.v0 = 1
neurons_output.tau = 5 * b2.ms
neurons_output.tau_t = 20 * b2.ms

# Set up monitors
spike_mon_input = b2.SpikeMonitor(neurons_input)
spike_mon_hidden = b2.SpikeMonitor(neurons_hidden)
spike_mon_output = b2.SpikeMonitor(neurons_output)

# Input to hidden layer synapses with dynamic STDP
synapses_input_hidden = b2.Synapses(
    neurons_input, neurons_hidden,
    model='w : 1',
    on_pre='v_post += w',
    on_post='w += 0.01 * (1 - w)'  # Dynamic weight update
)
synapses_input_hidden.connect(p=0.6)  # Increased connectivity
synapses_input_hidden.w = 'clip(0.2 * rand(), 0.1, 0.3)'  # Randomized initial weights

# Hidden to output layer synapses
synapses_hidden_output = b2.Synapses(
    neurons_hidden, neurons_output,
    model='w : 1',
    on_pre='v_post += w',
    on_post='w += 0.01 * (1 - w)'  # Dynamic weight update
)
synapses_hidden_output.connect(p=0.8)  # Higher connectivity to output
synapses_hidden_output.w = 'clip(0.2 * rand(), 0.1, 0.3)'  # Randomized initial weights

# Enhanced input encoding using rate coding
# Scale input currents based on pixel intensity for the first batch of test data
sample_data = X_test[0].reshape(-1)  # Take first test sample
neurons_input.input_value = sample_data  # Set input values
neurons_input.noise = 0.05 * np.random.randn(n_input_neurons)  # Set random noise

# Create and run network
network_improved = b2.Network(
    neurons_input, neurons_hidden, neurons_output,
    synapses_input_hidden, synapses_hidden_output,
    spike_mon_input, spike_mon_hidden, spike_mon_output
)
network_improved.run(500 * b2.ms)

improved_end = time.time()

# Process output spikes
spike_counts_output = np.bincount(spike_mon_output.i, minlength=n_output_neurons)
predicted_label_improved = np.argmax(spike_counts_output)
snn_predictions_improved = [predicted_label_improved for _ in range(len(y_test))]
snn_accuracy_improved = accuracy_score(np.argmax(y_test, axis=1), snn_predictions_improved)
improved_time = improved_end - improved_start

print(f"Improved SNN Accuracy: {snn_accuracy_improved:.4f}")
print(f"Improved SNN Simulation Time: {improved_time:.4f} seconds")

# --- Comparison of NN, Baseline SNN, and Improved SNN ---
print("\n--- Model Comparisons ---")
print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Baseline SNN Accuracy: {snn_accuracy_baseline:.4f}")
print(f"Improved SNN Accuracy: {snn_accuracy_improved:.4f}")
print(f"Baseline SNN vs NN Accuracy Gap: {nn_accuracy - snn_accuracy_baseline:.4f}")
print(f"Improved SNN vs NN Accuracy Gap: {nn_accuracy - snn_accuracy_improved:.4f}")
print(f"NN Memory Usage: {nn_mem_usage:.4f} MB")
print(f"NN Simulated 'Spike Activity': {nn_spike_like_metric:.4f}")

# Visualization of Results
plt.figure(figsize=(14, 10))

# Accuracy Comparison
plt.subplot(221)
plt.title("Model Accuracy Comparison")
models = ['NN', 'Baseline SNN', 'Improved SNN']
accuracies = [nn_accuracy, snn_accuracy_baseline, snn_accuracy_improved]
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Simulation Time Comparison
plt.subplot(222)
plt.title("Simulation Time Comparison")
times = [nn_time, baseline_time, improved_time]
plt.bar(models, times, color=['blue', 'green', 'red'])
plt.ylabel('Time (s)')

# Memory Usage Comparison
plt.subplot(223)
plt.title("NN Memory Usage")
plt.bar(['NN'], [nn_mem_usage], color=['blue'])
plt.ylabel('Memory (MB)')

# Spike Activity Comparison
plt.subplot(224)
plt.title("Simulated 'Spike Activity'")
plt.bar(['NN Spike Metric'], [nn_spike_like_metric], color=['blue'])
plt.ylabel('Spike Metric')

plt.tight_layout()
plt.show()

"""
# Third Iteration
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from memory_profiler import memory_usage
import time

# Fetch the MNIST dataset (handwritten digits)
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"]

# Preprocessing: Standardize pixel values and split into training and testing sets
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding for NN and maintain compatibility with SNN
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Traditional Neural Network (NN) Implementation
nn_start = time.time()
nn = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, activation='relu', solver='adam', learning_rate_init=0.001, random_state=42)
nn.fit(X_train, y_train)
nn_end = time.time()

# NN prediction and accuracy
y_pred_nn = nn.predict(X_test)
nn_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_nn, axis=1))
nn_time = nn_end - nn_start

# Calculate memory usage for NN
nn_start_mem = memory_usage()[0]
nn.predict(X_test)  # Simulate prediction to measure memory
nn_end_mem = memory_usage()[0]
nn_mem_usage = nn_end_mem - nn_start_mem

# Simulate spike activity metric for NN (as neuron activations)
nn_activations = np.mean(np.abs(nn.predict_proba(X_test)), axis=0)  # Average activation magnitude
nn_spike_like_metric = np.sum(nn_activations)

print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Traditional NN Training Time: {nn_time:.4f} seconds")
print(f"Traditional NN Memory Usage: {nn_mem_usage:.4f} MB")
print(f"Traditional NN Simulated 'Spike Activity' Metric: {nn_spike_like_metric:.4f}")

# Baseline SNN Implementation
n_neurons = 100
baseline_start = time.time()

# Define spiking neuron model for baseline
eqs_baseline = '''
dv/dt = (I-v)/tau : 1 (unless refractory)  # Membrane potential dynamics
I : 1  # Input current
tau : second  # Membrane time constant
'''

neurons_baseline = b2.NeuronGroup(
    n_neurons, eqs_baseline,
    threshold='v>1', reset='v=0', refractory=5*b2.ms
)
neurons_baseline.v = 'rand()'
neurons_baseline.I = 1.5
neurons_baseline.tau = 10 * b2.ms

# Monitor spikes for baseline
spike_mon_baseline = b2.SpikeMonitor(neurons_baseline)

# Synaptic connections for baseline
synapses_baseline = b2.Synapses(
    neurons_baseline, neurons_baseline,
    model='w : 1',
    on_pre='v_post += w',
    on_post='w -= 0.01'
)
synapses_baseline.connect(condition='i != j')
synapses_baseline.w = 0.1

# Run baseline network
network_baseline = b2.Network(neurons_baseline, spike_mon_baseline, synapses_baseline)
network_baseline.run(500 * b2.ms)

baseline_end = time.time()
spike_counts_baseline = np.bincount(spike_mon_baseline.i)
predicted_label_baseline = np.argmax(spike_counts_baseline)
snn_predictions_baseline = [predicted_label_baseline for _ in range(len(y_test))]
snn_accuracy_baseline = accuracy_score(np.argmax(y_test, axis=1), snn_predictions_baseline)
baseline_time = baseline_end - baseline_start

print(f"Baseline SNN Accuracy: {snn_accuracy_baseline:.4f}")
print(f"Baseline SNN Simulation Time: {baseline_time:.4f} seconds")

# --- Improved SNN Implementation ---
improved_start = time.time()

eqs_improved = '''
dv/dt = (I-v)/tau : 1 (unless refractory)
I : 1
tau : second
'''

neurons_improved = b2.NeuronGroup(
    n_neurons, eqs_improved,
    threshold='v>1', reset='v=0', refractory=4*b2.ms
)
neurons_improved.v = 'rand()'
neurons_improved.I = 1.6
neurons_improved.tau = 8 * b2.ms

spike_mon_improved = b2.SpikeMonitor(neurons_improved)
synapses_improved = b2.Synapses(
    neurons_improved, neurons_improved,
    model='w : 1',
    on_pre='v_post += w',
    on_post='w += 0.005'
)
synapses_improved.connect(condition='i != j')
synapses_improved.w = 0.15

network_improved = b2.Network(neurons_improved, spike_mon_improved, synapses_improved)
network_improved.run(500 * b2.ms)

improved_end = time.time()
spike_counts_improved = np.bincount(spike_mon_improved.i)
predicted_label_improved = np.argmax(spike_counts_improved)
snn_predictions_improved = [predicted_label_improved for _ in range(len(y_test))]
snn_accuracy_improved = accuracy_score(np.argmax(y_test, axis=1), snn_predictions_improved)
improved_time = improved_end - improved_start

print(f"Improved SNN Accuracy: {snn_accuracy_improved:.4f}")
print(f"Improved SNN Simulation Time: {improved_time:.4f} seconds")

# Comparison of NN, Baseline SNN, and Improved SNN
print("\n--- Model Comparisons ---")
print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Baseline SNN Accuracy: {snn_accuracy_baseline:.4f}")
print(f"Improved SNN Accuracy: {snn_accuracy_improved:.4f}")
print(f"Baseline SNN vs NN Accuracy Gap: {nn_accuracy - snn_accuracy_baseline:.4f}")
print(f"Improved SNN vs NN Accuracy Gap: {nn_accuracy - snn_accuracy_improved:.4f}")
print(f"NN Memory Usage: {nn_mem_usage:.4f} MB")
print(f"NN Simulated 'Spike Activity': {nn_spike_like_metric:.4f}")

# Visualization of Results
plt.figure(figsize=(14, 10))

# Accuracy Comparison
plt.subplot(221)
plt.title("Model Accuracy Comparison")
models = ['NN', 'Baseline SNN', 'Improved SNN']
accuracies = [nn_accuracy, snn_accuracy_baseline, snn_accuracy_improved]
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Simulation Time Comparison
plt.subplot(222)
plt.title("Simulation Time Comparison")
times = [nn_time, baseline_time, improved_time]
plt.bar(models, times, color=['blue', 'green', 'red'])
plt.ylabel('Time (s)')

# Memory Usage Comparison
plt.subplot(223)
plt.title("NN Memory Usage")
plt.bar(['NN'], [nn_mem_usage], color=['blue'])
plt.ylabel('Memory (MB)')

# Spike Activity Comparison
plt.subplot(224)
plt.title("Simulated 'Spike Activity'")
plt.bar(['NN Spike Metric'], [nn_spike_like_metric], color=['blue'])
plt.ylabel('Spike Metric')

plt.tight_layout()
plt.show()
"""
"""
# Here I am going to attach my an older iteration (second attempt) for comparison. This code only compares a baseline SNN and an Improved SNN
import brian2genn
import brian2 as b2
import memory_profiler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import time



# Fetch the MNIST dataset (handwritten digits)
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"]

# Preprocessing: Normalize pixel values and split into training and testing sets
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding for NN
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Implementing the Traditional Neural Network (NN) using scikit-learn
nn_start = time.time()
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, activation='relu', solver='adam', random_state=42)
nn.fit(X_train, y_train)
nn_end = time.time()

# NN prediction and accuracy
y_pred_nn = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_time = nn_end - nn_start
print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Traditional NN Training Time: {nn_time:.4f} seconds")

# START OF SNN IMPLEMENTATION
# Implementing the Spiking Neural Network (SNN) using Brian2 with GeNN backend for GPU support
n_neurons = 100
snn_start = time.time()

# Define spiking neuron model with increased input current and modified parameters
eqs = '''
dv/dt = (I-v)/tau : 1 (unless refractory)  # Membrane potential dynamics
I : 1  # Input current
tau : second  # Membrane time constant
'''

# Create neuron group with stronger input current
neurons = b2.NeuronGroup(n_neurons, eqs, threshold='v>1', reset='v=0', refractory=5*b2.ms)

# Initialize neuron parameters
neurons.v = 'rand()'
neurons.I = 1.5  # Increased input current to stimulate more spikes
neurons.tau = 10 * b2.ms  # Time constant for membrane potential decay

# Monitor spikes
spike_mon = b2.SpikeMonitor(neurons)

# Define synaptic connections with weight 'w' for STDP
synapses = b2.Synapses(neurons, neurons,
                       model='w : 1',  # Define synaptic weight variable
                       on_pre='v_post += w',  # Increase post-synaptic potential based on weight
                       on_post='w -= 0.01')  # STDP rule: decrease weight when post fires after pre
synapses.connect()
synapses.w = 0.1  # Initialize weights

# Create and run the network explicitly with GeNN backend
network = b2.Network(neurons, spike_mon, synapses)
network.run(500 * b2.ms)  # Simulation time

snn_end = time.time()
snn_time = snn_end - snn_start

# DECODING SPIKE ACTIVITY INTO CLASS PREDICTIONS - added on 11/29
# Count the spikes for each neuron (rate coding)
# Here, we assume that the neuron with the most spikes is the predicted class
spike_counts = np.bincount(spike_mon.i)
predicted_label = np.argmax(spike_counts)  # Neuron with most spikes is the prediction
snn_predictions = [predicted_label for _ in range(len(y_test))]  # Simplified prediction
snn_accuracy = accuracy_score(np.argmax(y_test, axis=1), snn_predictions)
print(f"Spiking Neural Network (SNN) Accuracy: {snn_accuracy:.4f}")

# PLOTTING: VISUALIZATION OF SPIKES AND NN ACCURACY
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Spiking Neural Network (SNN) Spikes")
plt.plot(spike_mon.t / b2.ms, spike_mon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

plt.subplot(122)
plt.title("Traditional Neural Network (NN) Accuracy")
plt.bar(['NN Accuracy'], [nn_accuracy], color=['blue'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# MEMORY USAGE PROFILING
nn_start_mem = memory_profiler.memory_usage()[0]
y_pred_nn = nn.predict(X_test)
nn_end_mem = memory_profiler.memory_usage()[0]
nn_mem_usage = nn_end_mem - nn_start_mem

snn_start_mem = memory_profiler.memory_usage()[0]
network.run(500 * b2.ms)  # Running the network again to measure memory usage
snn_end_mem = memory_profiler.memory_usage()[0]
snn_mem_usage = snn_end_mem - snn_start_mem

# Print results for comparison
print(f"Spiking Neural Network (SNN) Time: {snn_time:.4f} seconds")
print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Traditional NN Training Time: {nn_time:.4f} seconds")
print(f"Traditional NN Memory Usage: {nn_mem_usage:.4f} MB")
print(f"Spiking Neural Network (SNN) Simulation Time: {snn_time:.4f} seconds")
print(f"Spiking Neural Network (SNN) Memory Usage: {snn_mem_usage:.4f} MB")
"""

"""
# This was my first attempt
import brian2 as b2
import memory_profiler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import time

# Fetch the MNIST dataset (handwritten digits)
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"]

# Preprocessing: Normalize pixel values and split into training and testing sets
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Implementing the Traditional Neural Network (NN) using scikit-learn
nn_start = time.time()
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, activation='relu', solver='adam', random_state=42)
nn.fit(X_train, y_train)
nn_end = time.time()

# NN prediction and accuracy
y_pred_nn = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_time = nn_end - nn_start
print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Traditional NN Training Time: {nn_time:.4f} seconds")

# Implementing the Spiking Neural Network (SNN) using Brian2
n_neurons = 100
snn_start = time.time()

# Define spiking neuron model with increased input current and modified parameters
eqs = '''
dv/dt = (I-v)/tau : 1 (unless refractory)  # Membrane potential dynamics
I : 1  # Input current
tau : second  # Membrane time constant
'''

# Create neuron group with stronger input current
neurons = b2.NeuronGroup(n_neurons, eqs, threshold='v>1', reset='v=0', refractory=5*b2.ms)

# Initialize neuron parameters
neurons.v = 'rand()'
neurons.I = 1.5  # Increased input current to stimulate more spikes
neurons.tau = 10 * b2.ms  # Time constant for membrane potential decay

# Monitor spikes
spike_mon = b2.SpikeMonitor(neurons)

# Increase simulation time
simulation_time = 500 * b2.ms  # Run the simulation for 500 milliseconds
b2.run(simulation_time)

snn_end = time.time()
snn_time = snn_end - snn_start

# Visualization of spikes
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("Spiking Neural Network (SNN) Spikes")
plt.plot(spike_mon.t / b2.ms, spike_mon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

# Visualization of Traditional Neural Network (NN) Results
plt.subplot(122)
plt.title("Traditional Neural Network (NN) Accuracy")
plt.bar(['NN Accuracy'], [nn_accuracy], color=['blue'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')

# Show both plots
plt.tight_layout()
plt.show()

# Track memory usage for the NN
nn_start_mem = memory_profiler.memory_usage()[0]
y_pred_nn = nn.predict(X_test)
nn_end_mem = memory_profiler.memory_usage()[0]
nn_mem_usage = nn_end_mem - nn_start_mem

# Run the Spiking Neural Network (SNN) and measure time and memory usage
snn_start_mem = memory_profiler.memory_usage()[0]
b2.run(simulation_time)
snn_end_mem = memory_profiler.memory_usage()[0]
snn_mem_usage = snn_end_mem - snn_start_mem

# Print SNN time and comparison
print(f"Spiking Neural Network (SNN) Time: {snn_time:.4f} seconds")
print(f"Traditional NN Accuracy: {nn_accuracy:.4f}")
print(f"Traditional NN Training Time: {nn_time:.4f} seconds")
print(f"Traditional NN Memory Usage: {nn_mem_usage:.4f} MB")
print(f"Spiking Neural Network (SNN) Simulation Time: {snn_time:.4f} seconds")
print(f"Spiking Neural Network (SNN) Memory Usage: {snn_mem_usage:.4f} MB")
# Extra line to not stop at 666. 
"""
