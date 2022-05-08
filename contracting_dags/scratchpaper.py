import torch
import numpy as np
import tqdm
from network.dag import DAG
from data.xor import XORData
from data.aand import ANDData
from training.train_utils import train_epoch, compute_accuracy_and_atrition
from utils.torch_utils import to_float
from tools.visualize import draw_dag, draw_dag_weights

batch_size = 100

# dataset = XORData
dataset = ANDData

train_dataloader = torch.utils.data.DataLoader(dataset(1000), batch_size=batch_size, shuffle=True)
test_dataloader  = torch.utils.data.DataLoader(dataset(100),  batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape:  {train_labels.size()}"  )

x_all, _ = dataset.one_of_each()
dag = DAG(2, 1, 5)
print(dag)
draw_dag(dag)
print(dag(x_all))

torch.autograd.set_detect_anomaly(True)

def recreate_optimizer():
    return torch.optim.Adam(dag.parameters(), lr=0.01)
optimizer = recreate_optimizer()

accuracy = torch.zeros(1)
cooldown = 0
for epoch in tqdm.tqdm(range(1, 1001)):
    # Do!
    accuracy, atrition = compute_accuracy_and_atrition(dag, test_dataloader, accuracy)
    epoch_loss, regularizer_loss = train_epoch(dag, train_dataloader, optimizer, atrition=atrition)

    action_cooldown = 5
    threshold = 0.9
    structural_change = False
    if not cooldown and atrition > threshold:
        if dag.disable_weakest_node():
            cooldown = action_cooldown
            structural_change = True
    if not cooldown and atrition > threshold:
        if dag.disable_weakest_edge():
            cooldown = action_cooldown
            structural_change = True
    if not cooldown and atrition > threshold:
        if dag.contract_one_node_with_only_one_edge():
            structural_change = True
            cooldown = action_cooldown
    if structural_change:
        optimizer = recreate_optimizer()
    else:
        cooldown = max(0, cooldown - 1)

    # Report!
    draw_dag_weights(dag)
    to_print = [accuracy, atrition, to_float(regularizer_loss)]
    print(f"epoch {epoch}: {epoch_loss} @ {to_print}")
    print(dag(x_all))

draw_dag_weights(dag)
print(dag(x_all))
