import time
import numpy as np
import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.ao.quantization import get_default_qconfig_mapping
from torch.utils.data import Subset

# Set random seed for reproducibility
np.random.seed(0)

# ---------------------------
# 2. Data Preparation
# ---------------------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
full_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Select 1000 random indices from the full dataset
subset_indices = np.random.randint(0, len(full_dataset), 1000)
subset_dataset = Subset(full_dataset, subset_indices)
loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1, shuffle=False)

# ---------------------------
# 3. FP32 Inference (Baseline)
# ---------------------------
with torch.no_grad():
    # Warm-up run
    model(subset_dataset[0][0].unsqueeze(0))
    
answers_fp32 = []
start = time.time()
with torch.no_grad():
    for images, _ in loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        answers_fp32.append(predicted.item())
fp32_time = (time.time() - start) * 1000  # Convert seconds to milliseconds
print(f"FP32 Inference time: {fp32_time:.2f} ms")

# ---------------------------
# 4. Static Quantization using FX Graph Mode
# ---------------------------
# Preparation: use a sample image (here, full_dataset.data[0]) for tracing
qconfig_mapping = get_default_qconfig_mapping("x86")
model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, full_dataset.data[0])

# Calibration: run through the dataset to collect statistics
with torch.no_grad():
    for images, _ in loader:
        model_prepared(images)
# Convert to a quantized model
model_static_quantized = quantize_fx.convert_fx(model_prepared)
model_static_quantized.eval()

# Static quantization inference
with torch.no_grad():
    # Warm-up
    model_static_quantized(subset_dataset[0][0].unsqueeze(0))
    
answers_static = []
start = time.time()
with torch.no_grad():
    for images, _ in loader:
        outputs = model_static_quantized(images)
        _, predicted = torch.max(outputs, 1)
        answers_static.append(predicted.item())
static_time = (time.time() - start) * 1000
accuracy_static = sum(answers_fp32[i] == answers_static[i] for i in range(len(answers_fp32))) / len(answers_fp32)
print(f"Static Quantization Inference time: {static_time:.2f} ms")
print(f"Static Quantization Top-1 Accuracy: {accuracy_static:.3f}")

# ---------------------------
# 5. Dynamic Quantization
# ---------------------------
# Apply dynamic quantization to all Linear layers in the model.
# (Dynamic quantization will convert only modules specified in the mapping, here {torch.nn.Linear}.)
model_dynamic = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model_dynamic.eval()

# Dynamic quantization inference
with torch.no_grad():
    # Warm-up
    model_dynamic(subset_dataset[0][0].unsqueeze(0))
    
answers_dynamic = []
start = time.time()
with torch.no_grad():
    for images, _ in loader:
        outputs = model_dynamic(images)
        _, predicted = torch.max(outputs, 1)
        answers_dynamic.append(predicted.item())
dynamic_time = (time.time() - start) * 1000
accuracy_dynamic = sum(answers_fp32[i] == answers_dynamic[i] for i in range(len(answers_fp32))) / len(answers_fp32)
print(f"Dynamic Quantization Inference time: {dynamic_time:.2f} ms")
print(f"Dynamic Quantization Top-1 Accuracy: {accuracy_dynamic:.3f}")


