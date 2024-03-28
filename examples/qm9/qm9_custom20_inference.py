import os, json
import torch
import hydragnn
import pickle
from qm9_custom20_class import QM9_custom
import argparse
import random
#########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--pretraineddir", default='./logs', help="pretrained model directory")
parser.add_argument("--pretrainedmodelname", default='qm9_LOG2023_all20_1113', help="pretrained model name")
parser.add_argument("--graphfile", default='./data.pt',help="file of a graph object")
args = parser.parse_args()
modeldir = args.pretraineddir
model_name = args.pretrainedmodelname
graph_file = args.graphfile
#########################################################################################################
# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(modeldir,model_name, "config.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
if os.path.exists(graph_file):
    graphdata = torch.load(graph_file)
else:
    serial_data_name = "qm9_train_test_val_idx_lists.pkl"
    with open(os.path.join(os.path.dirname(__file__), "dataset", serial_data_name), "rb") as f:
        idx_train_list = pickle.load(f)
        idx_val_list = pickle.load(f)
        idx_test_list = pickle.load(f)

    minmax_file = os.path.join(os.path.dirname(__file__), "dataset", "train_minmax_output.pt")
    minmax_dict = torch.load(minmax_file)
    min_output_feature = minmax_dict["min_output_feature"]
    max_output_feature = minmax_dict["max_output_feature"]

    def qm9_pre_filter_test(data):
        return data.idx in idx_test_list
    test = QM9_custom(
        root=os.path.join(os.path.dirname(__file__), "dataset/all20/test"),
        var_config=var_config,
        pre_filter=qm9_pre_filter_test,
    )
    for data in test:
        num_nodes = data.x.size()[0]
        data.y[:-num_nodes, 0] = (
            data.y[:-num_nodes, 0] - min_output_feature[:-1, 0]
        ) / (max_output_feature[:-1, 0] - min_output_feature[:-1, 0])
        data.y[-num_nodes:, 0] = (
            data.y[-num_nodes:, 0] - min_output_feature[-1, 0]
        ) / (max_output_feature[-1, 0] - min_output_feature[-1, 0])
    graphdata=test[random.randint(0, len(test))]
    torch.save(graphdata, graph_file)
##################################################################################################################
#load pretrained model
model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)
hydragnn.utils.model.load_existing_model(model, model_name, path=modeldir)
model.eval()
##################################################################################################################
print(graphdata)
pred = model(graphdata)
ytrue = graphdata.y
pred = model(graphdata)
print("Head, Normalized properties, True  VS   Pred")
for ival in range(len(ytrue)):
    if ival<19:
        ihead = ival
        print("%4d, %21s, %.3f VS  %.3f"%(ival, var_config["output_names"][ihead], pred[ival].item(), ytrue[ival].item()))
    else:
        ihead = 19
        print("%4d, %21s, atom %2d,  %.3f VS %.3f"%(ihead, var_config["output_names"][ihead], ival-ihead, pred[ihead][ival-ihead].item(), ytrue[ival].item()))

    
