# LSTM-based-Neural-Decoding-of-non-Human-Primates
Implementation of the IEEE BioCAS 2024 Grand Challenge focused on advancing neural decoding techniques for motor control in non-human primates using LSTMs.

# Dataset
Original dataset - https://zenodo.org/records/583331

The above link is for reference, the actual loading and processing are done using the neurobench code harness
- Neurobench code harness - https://github.com/NeuroBench/neurobench
- We only use neurobench for loading/processing the data and benchmarking on the final test set. Model training is done from scratch.

# Running 
- Follow the steps in the neurobench repo to setup the neurobench environment
- Run load_data.py from the neurobench env to load and preprocess the data
