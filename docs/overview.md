FedCNC – Federated Multi-Task Learning for Predictive Maintenance in CNC Machines

FedCNC is a federated learning–based multi-task deep learning system designed for predictive maintenance in CNC machines.
The framework trains a shared model across multiple clients without exchanging raw data.
It predicts:

Tool wear progression

Tool failure (TWF)

Machine failure

Early failure warnings (preventive alerts)

Features

Federated learning with decentralized training

Multi-task neural network (predicts 3 targets simultaneously)

Privacy-preserving (raw CNC data never leaves clients)

Scalable to multiple CNC machines

Early warning system with failure horizon estimation

Supports industrial CNC predictive maintenance datasets (AI4I dataset)

Project Structure
FedCNC/
│── client.py
│── server.py
│── model.py
│── prepare_split.py
│── utils.py
│── generate_data.py
│── data/
│     ├── clients/
│     ├── mapping.json
│── saved_models/
│── README.md
│── LICENSE

For accessing my whole project with my model => " https://drive.google.com/drive/folders/1fcB5fF8G63d6XgtAyjMQYCb57mnGxQGM?usp=sharing "  Access this link. 
**Note => used codes are not optimized
This model tarin or develope for Research Purpose**

Dataset
This project uses the "AI4I Predictive Maintenance CNC Dataset":

https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

Dataset columns used:

Air temperature [K]

Process temperature [K]

Rotational speed [rpm]

Torque [Nm]

Tool wear [min]

Machine failure

TWF

HDF, PWF, OSF, RNF

Product ID (used as tool ID)

Installation
1. Clone repository
git clone https://github.com/yourname/FedCNC.git
cd FedCNC

2. Create virtual environment
python -m venv venv

3. Activate environment
venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt


If no requirements file exists:

pip install flwr torch numpy pandas scikit-learn

Preparing Dataset

Place your dataset as:

cnc_dataset.csv


Run the splitter:

python prepare_split.py --input cnc_dataset.csv --output data/clients --num-clients 4


Output structure:

data/clients/
    client_0.csv
    client_1.csv
    client_2.csv
    client_3.csv
    mapping.json

Running the Server

Start the federated server:

python server.py --num-rounds 500 --data-dir data/clients --mapping data/clients/mapping.json


The server performs:

Model aggregation

Validation

Failure trend analysis

Early warning generation

Failure horizon prediction

Running Clients

Each client must be started in a separate terminal.

Client 0:
python client.py --client-id 0 --data-dir data/clients --mapping data/clients/mapping.json

Client 1:
python client.py --client-id 1 --data-dir data/clients --mapping data/clients/mapping.json

Client 2:
python client.py --client-id 2 --data-dir data/clients --mapping data/clients/mapping.json

Client 3:
python client.py --client-id 3 --data-dir data/clients --mapping data/clients/mapping.json

Model Architecture

A multi-task neural network is used:

Shared dense layers

Three output heads:

Tool Wear Class (Softmax)

Machine Failure (Softmax)

Tool Failure (Softmax)

Loss Function:

Total Loss = L_toolwear + L_machinefailure + L_toolfailure


Optimizer:

Adam (lr = 1e-3)

Expected Output
Server Output

Round-by-round training loss

Accuracy for 3 tasks

Aggregated metrics

Early failure warnings:

Example:

EARLY WARNING:
Failure probability rising.
Estimated failure in next 8 rounds.
Suggested Action: Inspect tool, reduce spindle load.

Client Output

Local training loss

Tool wear accuracy

Machine failure accuracy

Tool failure accuracy

Example:

Loss: 0.317
Accuracies → Tool: 91% | Machine: 94% | ToolFail: 87%

Why Federated Learning for CNC Machines

Keeps industrial machine data private

Avoids risk of factory data leakage

Allows multi-machine learning without merging datasets

Enables continuous improvement using distributed CNC data

Citation
@misc{fedcnc2025,
  title={Federated Multi-Task Learning for Predictive Maintenance in CNC Machines},
  author={Your Name},
  year={2025},
  howpublished={GitHub Repository}
}
