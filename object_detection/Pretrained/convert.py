import torch

checkpoint = torch.load('checkpoint-309.pth.tar', map_location='cpu')
model = checkpoint['state_dict']
torch.save(model, 'wavevit_b.pth')

