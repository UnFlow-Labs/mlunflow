from pydantic import BaseModel

from unflow.core.unflow_core import unflowdecorator


class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    shuffle: bool
    num_workers: int

class Config(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int
    data_config: DataConfig
    optimizer: str
    optimizer_params: dict

@unflowdecorator()
def train_model(config: Config):
    # Simulate training process
    print(f"Training model with config: {config}")
    return {"status": "success", "final_loss": 0.1234}

config1 = Config(
    learning_rate=0.01,
    batch_size=32,
    num_epochs=10,
    data_config=DataConfig(
        data_path="/path/to/data1",
        batch_size=32,
        shuffle=True,
        num_workers=4
    ),
    optimizer="adam",
    optimizer_params={"weight_decay": 0.01}
)
#change in config1
config2 = Config(
    learning_rate=0.01,
    batch_size=32,
    num_epochs=10,
    data_config=DataConfig(
        data_path="/path/to/data2",  # Changed data path
        batch_size=32,
        shuffle=True,
        num_workers=4
    ),
    optimizer="adam",
    optimizer_params={"weight_decay": 0.01}
)
train_model.clear_graph()
train_model(config1)
train_model(config2)
#get edges in the graph
g = train_model.compute_graph.graph
print(f"Graph has {len(g.nodes)} nodes and {len(g.edges)} edges.")
for u, v, data in g.edges(data=True):
    print(f"Edge from {u} to {v} with transformation: {data['transformation'].args_changes}")
