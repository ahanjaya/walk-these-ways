import torch

class FeasibilityNet(torch.nn.Module):
    def __init__(self, num_feasibility_obs):
        super(FeasibilityNet, self).__init__()
        
        # Feasibility Network
        feasibility_hidden_activation = torch.nn.LeakyReLU()
        feasibility_output_activation = torch.nn.Sigmoid()
        feasibility_hidden_dims = [128, 128]

        feasibility_layers = []
        feasibility_layers.append(
            torch.nn.Linear(num_feasibility_obs, feasibility_hidden_dims[0])
        )
        feasibility_layers.append(feasibility_hidden_activation)

        for l in range(len(feasibility_hidden_dims)):
            if l == len(feasibility_hidden_dims) - 1:
                feasibility_layers.append(
                    torch.nn.Linear(feasibility_hidden_dims[l], 1)
                )
                feasibility_layers.append(feasibility_output_activation)
            else:
                feasibility_layers.append(
                    torch.nn.Linear(
                        feasibility_hidden_dims[l],
                        feasibility_hidden_dims[l + 1],
                    )
                )
                feasibility_layers.append(feasibility_hidden_activation)
        self.feasibility_module = torch.nn.Sequential(*feasibility_layers)
        
        print(f"Feasibility Module: {self.feasibility_module}")

    def forward(self, x):
        return self.feasibility_module(x)

    def act_inference(self, x):
        with torch.no_grad():
            return self.forward(x)
