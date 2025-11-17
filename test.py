import torch

# Assuming MyModel is the class definition of your saved model
from my_model_module import MyModel 

# Load the entire model
loaded_model = torch.load('model.pth') 

# Set the model to evaluation mode if you're using it for inference
loaded_model.eval() 