from examples.LeNetMNIST.lenet import DataConfig, ModelConfig, TrainingConfig, run_experiment

# run_experiment.clear_graph()  # Clear the graph before starting the experiment
training_config = TrainingConfig(num_epochs=5, optimizer="Adam", optimizer_params={"lr": 0.001})
result = run_experiment(training_config)
print(f"Final Result: Loss = {result['loss']:.4f}, Accuracy = {result['accuracy']:.2f}%")

#Hypothesis0: AdamW optimizer might perform better than Adam in this case due to its decoupled weight decay, 
# which can help in regularizing the model and potentially improving generalization.
new_training_config = TrainingConfig(num_epochs=5, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01})
new_result = run_experiment(new_training_config)
print(f"New Result with AdamW: Loss = {new_result['loss']:.4f}, Accuracy = {new_result['accuracy']:.2f}%")

#Hypothesis1: Increasing the number of epochs might lead to better convergence and potentially improved accuracy, but it also increases the risk of overfitting if the model is trained for too long without proper regularization.
new_training_config_longer_epochs = TrainingConfig(num_epochs=10, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01})
new_result_longer_epochs = run_experiment(new_training_config_longer_epochs)
print(f"New Result with AdamW and longer epochs: Loss = {new_result_longer_epochs['loss']:.4f}, Accuracy = {new_result_longer_epochs['accuracy']:.2f}%")

#Hypothesis2: Using a different optimizer like SGD with momentum might lead to different convergence behavior 
# and potentially better generalization, especially if the learning rate is tuned appropriately.

new_training_config_sgd = TrainingConfig(num_epochs=10, optimizer="SGD", optimizer_params={"lr": 0.01, "momentum": 0.9})
new_result_sgd = run_experiment(new_training_config_sgd)
print(f"New Result with SGD: Loss = {new_result_sgd['loss']:.4f}, Accuracy = {new_result_sgd['accuracy']:.2f}%")

#Hypothesis3: using a different activation function like LeakyReLU might help in mitigating the vanishing gradient problem and could potentially lead to better convergence and improved accuracy, especially in deeper networks.
new_training_config_leakyrelu = TrainingConfig(num_epochs=10, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01},
                                              model_settings=ModelConfig(use_batchnorm=True, dropout_rate=0.5,
                                                                          n_blocks=2, block_channels=[6, 16], kernel_sizes=[5, 5],
                                                strides=[1, 1], paddings=[0, 0], n_fc_layers=2, fc_hidden_sizes=[120, 84], activation_fn="LeakyReLU"))
new_result_leakyrelu = run_experiment(new_training_config_leakyrelu)
print(f"New Result with LeakyReLU: Loss = {new_result_leakyrelu['loss']:.4f}, Accuracy = {new_result_leakyrelu['accuracy']:.2f}%")

#RepeatedHypothesis3: using a different activation function like LeakyReLU might help in mitigating the vanishing gradient problem and could potentially lead to better convergence and improved accuracy, especially in deeper networks.
new_training_config_leakyrelu = TrainingConfig(num_epochs=10, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01},
                                              model_settings=ModelConfig(use_batchnorm=True, dropout_rate=0.5,
                                                                          n_blocks=2, block_channels=[6, 16], kernel_sizes=[5, 5],
                                                strides=[1, 1], paddings=[0, 0], n_fc_layers=2, fc_hidden_sizes=[120, 84], activation_fn="LeakyReLU"))
new_result_leakyrelu = run_experiment(new_training_config_leakyrelu)
if new_result_leakyrelu is not None:
    print(f"New Result with LeakyReLU: Loss = {new_result_leakyrelu['loss']:.4f}, Accuracy = {new_result_leakyrelu['accuracy']:.2f}%")
else:
    print("New Result with LeakyReLU: Experiment was run before.")

#Hypothesis4: using a different activation function like ELU might help in mitigating the vanishing gradient problem and could potentially lead to better convergence and improved accuracy, especially in deeper networks.
new_training_config_elu = TrainingConfig(num_epochs=10, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01},
                                              model_settings=ModelConfig(use_batchnorm=True, dropout_rate=0.5,
                                                                          n_blocks=2, block_channels=[6, 16], kernel_sizes=[5, 5],
                                                strides=[1, 1], paddings=[0, 0], n_fc_layers=2, fc_hidden_sizes=[120, 84], activation_fn="ELU"))
new_result_elu = run_experiment(new_training_config_elu)
print(f"New Result with ELU: Loss = {new_result_elu['loss']:.4f}, Accuracy = {new_result_elu['accuracy']:.2f}%")

#Hypothesis5: using tanh activation function might lead to different convergence behavior and could potentially improve accuracy, especially if the model is sensitive to the choice of activation function
new_training_config_tanh = TrainingConfig(num_epochs=10, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01},
                                              model_settings=ModelConfig(use_batchnorm=True, dropout_rate=0.5,
                                                                          n_blocks=2, block_channels=[6, 16], kernel_sizes=[5, 5],
                                                strides=[1, 1], paddings=[0, 0], n_fc_layers=2, fc_hidden_sizes=[120, 84], activation_fn="Tanh"))
new_result_tanh = run_experiment(new_training_config_tanh)
print(f"New Result with Tanh: Loss = {new_result_tanh['loss']:.4f}, Accuracy = {new_result_tanh['accuracy']:.2f}%")

#Hypothesis6: using Muon optimizer might lead to different convergence behavior and could potentially improve accuracy, especially if the model is sensitive to the choice of optimizer.
new_training_config_muon = TrainingConfig(num_epochs=10, optimizer="Muon", optimizer_params={"lr": 0.001, "weight_decay": 0.01})
new_result_muon = run_experiment(new_training_config_muon)
if new_result_muon is not None:
    print(f"New Result with Muon: Loss = {new_result_muon['loss']:.4f}, Accuracy = {new_result_muon['accuracy']:.2f}%")
else:
    print("New Result with Muon: Experiment has failed or was run before and did not complete successfully.")

#Hypothesis7: test different batch sizes might lead to different convergence behavior and could potentially improve accuracy, especially if the model is sensitive to the choice of batch size.
for batch_size in [32, 64, 128, 1024, 2048]:
    new_training_config_batchsize = TrainingConfig(num_epochs=10, optimizer="AdamW", optimizer_params={"lr": 0.001, "weight_decay": 0.01},
                                               model_settings=ModelConfig(use_batchnorm=True, dropout_rate=0.5,
                                                                           n_blocks=2, block_channels=[6, 16], kernel_sizes=[5, 5],
                                                 strides=[1, 1], paddings=[0, 0], n_fc_layers=2, fc_hidden_sizes=[120, 84]), 
                                                 data_settings=DataConfig(batch_size=batch_size))
    new_result_batchsize = run_experiment(new_training_config_batchsize)
    print(f"New Result with batch size {batch_size}: Loss = {new_result_batchsize['loss']:.4f}, Accuracy = {new_result_batchsize['accuracy']:.2f}%")

