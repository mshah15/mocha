using CSV
using HDF5

train_data_name = "cc_train"
test_data_name = "cc_test"

cc_data = CSV.read("../data/cc_data.csv",nullable=false)
cc_data = convert(Array{Any,2},cc_data[2:30001,:])

cc_data = map(x->parse(Float32,x),cc_data)
cc_data = convert(Array{Float32,2},cc_data)
cc_data = cc_data[shuffle(1:end),:]


cc_train_features = cc_data[1:24000,1:24]
cc_train_labels = cc_data[1:24000,25]

cc_test_features = cc_data[24001:end,1:24]
cc_test_labels = cc_data[24001:end,25]

cc_train_features = transpose(cc_train_features)

cc_train_labels = transpose(cc_train_labels)
cc_train_labels = convert(Array{Float32,2},cc_train_labels)

cc_test_features = transpose(cc_test_features)

cc_test_labels = transpose(cc_test_labels)
cc_test_labels = convert(Array{Float32,2},cc_test_labels)

h5open("../data/$train_data_name.hdf5", "w") do h5
  write(h5, "data", cc_train_features[:,:])
  write(h5, "label", cc_train_labels[:])
end 

h5open("../data/$test_data_name.hdf5", "w") do h5
  write(h5, "data", cc_test_features[:,:])
  write(h5, "label", cc_test_labels[:])
end 

# Generates the necessary .txt file
run(pipeline(Cmd(`echo ../data/$(train_data_name).hdf5`), stdout="../data/txt/$train_data_name.txt", stderr="errs.txt"))
run(pipeline(Cmd(`echo ../data/$(test_data_name).hdf5`), stdout="../data/txt/$test_data_name.txt", stderr="errs.txt"))

