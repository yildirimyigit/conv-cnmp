
import sys
from model.conv_cnmp import ConvCNMP
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
                
if len(sys.argv) != 2:
    raise Exception("Run type (train/test) is missing")
if sys.argv[1] != "train" and sys.argv[1] != "test":
    raise Exception("Run type must be either train or test: %s is found" % sys.argv[1])
else:
    run_type = sys.argv[1]

if run_type == "train":
    train_model = ConvCNMP().to(device)
    optimizer = torch.optim.Adam(lr=1e-4, params=train_model.parameters())
    
else:
    test_model = ConvCNMP().to(device)


