import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"模型已加载自 {model_path}")