import torch
import numpy as np
import tqdm
from network.dag import DAG
from data.xor import XORData
from training.train_utils import train_epoch, compute_accuracy_and_atrition, to_float
from tools.visualize import draw_dag, draw_dag_weights

batch_size = 100
train_dataloader = torch.utils.data.DataLoader(XORData(1000), batch_size=batch_size, shuffle=True)
test_dataloader  = torch.utils.data.DataLoader(XORData(100),  batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape:  {train_labels.size()}"  )

x_all, _ = XORData.one_of_each()
dag = DAG(2, 1, 5)
print(dag)
draw_dag(dag)
print(dag(x_all))

torch.autograd.set_detect_anomaly(True)

def recreate_optimizer():
    return torch.optim.Adam(dag.parameters(), lr=0.1)
optimizer = recreate_optimizer()

accuracy = torch.zeros(1)
for epoch in tqdm.tqdm(range(1, 101)):
    # Do!
    accuracy, atrition = compute_accuracy_and_atrition(dag, test_dataloader, accuracy)
    epoch_loss, regularizer_loss = train_epoch(dag, train_dataloader, optimizer, atrition=atrition)

    if epoch % 10 == 0 and len(dag.hidden_nodes) > 2:
        new_params = dag.disable_weakest_node()
        optimizer = recreate_optimizer()

    # Report!
    draw_dag_weights(dag)
    to_print = [accuracy, atrition, to_float(regularizer_loss)]
    print(f"epoch {epoch}: {epoch_loss} @ {to_print}")

draw_dag_weights(dag)
print(dag(x_all))
