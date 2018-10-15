using Mocha

train_filename = "cc_train.txt"
test_filename = "cc_test.txt"
exp_dir = "snapshots/cc_net/"

# Convenience for training network multiple times from different sources.
if length(ARGS) > 0
    train_filename = ARG2[1]
    test_filename = ARGS[2]
    exp_dir = ARGS[3]
end

# Input
data_layer  = HDF5DataLayer(name="train-data", source="cc_data/txt/$train_filename",
batch_size=64, shuffle=true)

#data_tt_layer  = HDF5DataLayer(name="train-data", source="data/train.txt",
#batch_size=64, shuffle=true)

# Inner product layer, input is determined by the "bottoms" option,
#   in this case, the data layer

fc1_layer  = InnerProductLayer(name="ip1", output_dim=256,
neuron=Neurons.ReLU(), bottoms=[:data], tops=[:ip1])

fc2_layer  = InnerProductLayer(name="ip2", output_dim=256,
neuron=Neurons.ReLU(), bottoms=[:ip1], tops=[:ip2])

fc3_layer  = InnerProductLayer(name="ip3", output_dim=64,
neuron=Neurons.ReLU(), bottoms=[:ip2], tops=[:ip3])

fc4_layer  = InnerProductLayer(name="ip4", output_dim=64,
neuron=Neurons.Sigmoid(), bottoms=[:ip3], tops=[:ip4])

fc5_layer  = InnerProductLayer(name="ip5", output_dim=32,
neuron=Neurons.Sigmoid(), bottoms=[:ip4], tops=[:ip5])

#=
    Output Dim is 2 because this is the final layer, so 1 or 0
    this is the classification layer
=#


fc6_layer  = InnerProductLayer(name="ip6", output_dim=2,
bottoms=[:ip5], tops=[:ip6])



drop_input  = DropoutLayer(name="drop_in", bottoms=[:data], ratio=0.2)
drop_fc1 = DropoutLayer(name="drop_fc1", bottoms=[:ip1], ratio=0.5)

#= 
    Loss layer -- connected to the second IP layer and "label" from
        the data layer.

=#
loss_layer = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])
# loss_layer = BinaryCrossEntropyLossLayer(name="loss", bottoms=[:ip2, :label])

# Configure and build
backend = CPUBackend()
init(backend)

# Putting the network together
common_layers = [fc1_layer, fc2_layer]
drop_layers = [drop_input, drop_fc1]

net = Net("cc-train", backend, [data_layer, common_layers..., drop_layers..., loss_layer])

# Setting up the solver, this is identical to the MNIST tutorial

method = SGD()
params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.03, 0.0001, 0.75),
    load_from=exp_dir)


solver = Solver(method, params)


#=
      "This sets up the coffee lounge, which holds data reported during coffee breaks. 
      Here we also specify a file to save the information we accumulated in coffee 
      breaks to disk. Depending on the coffee breaks, useful statistics such as 
      objective function values during training will be saved, and can be loaded later 
      for plotting or inspecting."
=#

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)


#=
    "First, we allow the solver to have a coffee break after every 100 iterations 
    so that it can give us a brief summary of the training process. By default 
    TrainingSummary will print the loss function value on the last training mini-batch."
=#

add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# Snapshot
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=500)

#= 
    Evaluation network. Run against the test set
=#

data_layer_test = HDF5DataLayer(name="test-data", source="cc_data/txt/$test_filename", batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
test_net = Net("cc-test", backend, [data_layer_test, common_layers..., drop_layers..., acc_layer])



add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

@time solve(solver, net)




destroy(net)
destroy(test_net)
shutdown(backend)
