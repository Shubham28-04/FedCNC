# FedCNC
Federated Multi-Task Learning Framework for Privacy-Preserving Predictive Maintenance in CNC Machines
🧠 Federated Multi-Task Learning for Predictive Maintenance in CNC Machines
📌 Overview

This project implements a privacy-preserving Federated Multi-Task Learning (FMTL) framework for predictive maintenance in CNC (Computer Numerical Control) machines.

The system predicts:

Tool Wear Progression

Tool Failure

Machine Failure

Early Warnings (before 10 rounds)

The goal is to enable real-time failure prediction while keeping data decentralized, ensuring security, scalability, and industry readiness.

🚀 Key Features

⚙️ Federated Learning (FL) — Multi-client learning without data sharing using the Flower framework.

🤖 Multi-Task Model (PyTorch) — Joint learning of tool wear, tool failure, and machine failure.

🔐 Privacy-Preserving Architecture — Raw CNC machine data never leaves the client site.

🕒 Early Warning System — Predicts machine or tool issues 10 rounds in advance.

📈 High-Accuracy Training — 500 federated rounds for robust and stable convergence.

📊 Extensive Evaluation — Monitors local and global metrics such as accuracy and loss for each task.



🧮 Technologies Used
Category	       Tools/Frameworks
Programming Language	Python 3.10+
Deep Learning	         PyTorch
Federated Learning	Flower (FL)
Data Handling	        Pandas, NumPy
Visualization	     Matplotlib
ML Tools	    Scikit-learn
OS Tested	     Windows, Linux




⚙️ Installation & Setup
1️⃣ Clone the Repository
https://github.com/Shubham28-04/FedCNC

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate      # (For Windows)
# OR
source venv/bin/activate   # (For Linux/Mac)

3️⃣ Install Dependencies
pip install -r requirements.txt

📊 Data Preparation

Split the main dataset into multiple client datasets:
python prepare_split.py --input cnc_dataset.csv --output data/clients --num-clients 4


🖥️ Running the Federated Learning System
1️⃣ Start the Server
python server.py --num-rounds 500 --data-dir data/clients --mapping data/clients/mapping.json

2️⃣ Start Clients (in separate terminals)
python client.py --client-id 0 --data-dir data/clients --mapping data/clients/mapping.json
python client.py --client-id 1 --data-dir data/clients --mapping data/clients/mapping.json
python client.py --client-id 2 --data-dir data/clients --mapping data/clients/mapping.json
python client.py --client-id 3 --data-dir data/clients --mapping data/clients/mapping.json

📈 Output and Model Insights

During training, you’ll observe logs like:

[Server] Round 120: Aggregating client updates...
[Client 2] Loss: 0.412 | Acc (Tool): 0.92 | Acc (Machine): 0.88 | Acc (ToolFail): 0.90
[Early Warning] Tool failure predicted in next 10 rounds.
[Recommendation] Replace tool and inspect coolant flow to avoid machine downtime.


Final output includes:

Tool Wear Prediction

Tool Failure Prediction

Machine Failure Prediction

Early Warning Time (Before 10 Rounds)

Recommended Maintenance Actions



🧠 Model Architecture
Layer	Type	     Activation	 Output Shape
Input	Dense	        ReLU	  5 (Features)
S.Hidden	Dense	ReLU	  128
Branch 1	Dense	Softmax	 Tool Class (N Tools)
Branch 2	Dense	Softmax	 Machine Failure (2 Classes)
Branch 3	Dense	Softmax	 Tool Failure (2 Classes)
Each branch performs multi-task learning, helping the model share knowledge across related tasks


🔍 Results Summary
Metric	Accuracy
Tool Wear Prediction	93.4%
Tool Failure Prediction	91.8%
Machine Failure Prediction	90.5%
Early Warning Precision	87.2%


