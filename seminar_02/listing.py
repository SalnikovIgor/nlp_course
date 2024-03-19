# 1. Create and activate venv
# python3 -m venv venv
# source venv/bin/activate

# 2. Install requirements
# pip install -r requirements.txt

# 3. Import torch
import torch

# 4. Create tensor
a = torch.tensor([[1, 2], [2, 4]])

# 5. Check shape
print(a.shape)
print(a.size())

# 6. Check tensor
print(a)
print(a.get_device())  # -1 for CPU

# 7. Check CUDA device
print(torch.cuda.is_available())

# 8. Create CUDA tensor
b = torch.tensor([[1, 2], [2, 4]], device='cuda')
print(b)
print(b.get_device())

# 9. Sum a and b
try:
    print(a+b)
except Exception as e:
    print(f'ERROR: {e}')

# 10. Move a to cuda
c = a.to('cuda')
print(b + c)
print(b - c)
print(b * c)
print(b / c)

# 11. Help function
print(torch.rand(1, 2, 3))
print(torch.ones(1, 2, 3))
print(torch.zeros(1, 2, 3))
print()

# 12. Layers (slide)
liner = torch.nn.Linear(2, 10, bias=False)

d = a.to(torch.float)
w = liner.weight.clone().detach()

print('Layer weight:')
print(liner.weight)
print()
print(liner(d))
print(torch.matmul(d, w.T))

# 13. Layers (slide)
print(a.shape)
embedding = torch.nn.Embedding(100, 10)
print(embedding(a))
print(embedding(a).shape)

# 13. RNN (slide)
rnn = torch.nn.RNN(15, 21, 3)
input = torch.randn(5, 3, 15)
h0 = torch.randn(3, 3, 21)
output, hn = rnn(input, h0)

print(output)
print(output.shape)
