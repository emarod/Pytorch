import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP1(nn.Module):
  def __init__(self):
    super(MLP1, self).__init__()
  
    DROPOUT_RATE = 0.5 
    # Capas Lineales
    self.l1 = nn.Linear(3072, 512)
    self.l2 = nn.Linear(512, 256)
    self.l3 = nn.Linear(256, 128)
    self.l4 = nn.Linear(128, 10)
    # Capa de Dropout
    self.dropout = nn.Dropout(DROPOUT_RATE)

  def forward(self, x):
    out = self.l1(x)
    out = F.relu(out)
    out = self.dropout(out)

    out = self.l2(out)
    out = F.relu(out)
    out = self.dropout(out)

    out = self.l3(out)
    out = F.relu(out)
    out = self.dropout(out)

    out = self.l4(out)

    return out

class CNN1(nn.Module):
  def __init__(self):
    super(CNN1, self).__init__()

    # --- PARÁMETROS DE REGULARIZACIÓN ---
    DROPOUT_RATE_CONV = 0.1
    DROPOUT_RATE_FC = 0.5

    # --- BLOCK 1 ---
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(32) 
    self.dropout1 = nn.Dropout2d(DROPOUT_RATE_CONV)

    # --- BLOCK 2 ---
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) 
    self.bn2 = nn.BatchNorm2d(64)
    self.dropout2 = nn.Dropout2d(DROPOUT_RATE_CONV)

    # --- BLOCK 3 ---
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.bn3 = nn.BatchNorm2d(128)
    self.dropout3 = nn.Dropout2d(DROPOUT_RATE_CONV)

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.l1 = nn.Linear(in_features=128 * 2 * 2, out_features=128)
    self.bn_fc1 = nn.BatchNorm1d(128)
    self.dropout_fc = nn.Dropout(DROPOUT_RATE_FC) 
    self.l2 = nn.Linear(in_features=128, out_features=10)

  def forward(self, x):
    # --- BLOCK 1: Conv -> BN -> ReLU -> Dropout -> Pool ---
    out = self.conv1(x)
    out = self.bn1(out) 
    out = F.relu(out)
    out = self.dropout1(out) # Aplicar Dropout 2D
    out = self.pool(out) # Salida: (Batch, 32, 16, 16)
    
    # --- BLOCK 2: Conv -> BN -> ReLU -> Dropout -> Pool ---
    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out)
    out = self.dropout2(out)
    out = self.pool(out) # Salida: (Batch, 64, 7, 7)
    
    # --- BLOCK 3: Conv -> BN -> ReLU -> Dropout -> Pool ---
    out = self.conv3(out)
    out = self.bn3(out)
    out = F.relu(out)
    out = self.dropout3(out)
    out = self.pool(out) # Salida: (Batch, 128, 2, 2)

    # Aplanar
    out = torch.flatten(out, 1) # Salida: (Batch, 512)

    # --- CAPA LINEAL 1: Linear -> BN -> ReLU -> Dropout ---
    out = self.l1(out)
    out = self.bn_fc1(out)
    out = F.relu(out)
    out = self.dropout_fc(out) # Aplicar Dropout

    # --- CAPA LINEAL 2: Salida Final ---
    out = self.l2(out)
    
    return out
  
class CNN2(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(CNN1, self).__init__()

    # --- PARÁMETROS DE REGULARIZACIÓN ---
    DROPOUT_RATE_CONV = 0.1
    DROPOUT_RATE_FC = 0.5

    # --- BLOCK 1 ---
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(32) 
    self.dropout1 = nn.Dropout2d(DROPOUT_RATE_CONV)

    # --- BLOCK 2 ---
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) 
    self.bn2 = nn.BatchNorm2d(64)
    self.dropout2 = nn.Dropout2d(DROPOUT_RATE_CONV)

    # --- BLOCK 3 ---
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.bn3 = nn.BatchNorm2d(128)
    self.dropout3 = nn.Dropout2d(DROPOUT_RATE_CONV)

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.l1 = nn.Linear(in_features=128 * 2 * 2, out_features=10)

  def forward(self, x):
    # --- BLOCK 1: Conv -> BN -> ReLU -> Dropout -> Pool ---
    out = self.conv1(x)
    out = self.bn1(out) 
    out = F.relu(out)
    out = self.dropout1(out) # Aplicar Dropout 2D
    out = self.pool(out) # Salida: (Batch, 32, 16, 16)
    
    # --- BLOCK 2: Conv -> BN -> ReLU -> Dropout -> Pool ---
    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out)
    out = self.dropout2(out)
    out = self.pool(out) # Salida: (Batch, 64, 7, 7)
    
    # --- BLOCK 3: Conv -> BN -> ReLU -> Dropout -> Pool ---
    out = self.conv3(out)
    out = self.bn3(out)
    out = F.relu(out)
    out = self.dropout3(out)
    out = self.pool(out) # Salida: (Batch, 128, 2, 2)

    # Aplanar
    out = torch.flatten(out, 1) # Salida: (Batch, 512)

    # --- CAPA LINEAL 1: Linear -> BN -> ReLU -> Dropout ---
    out = self.l1(out)

    return out
  
class CNN3(nn.Module):
  # La firma del constructor se mantiene simple
  def __init__(self): 
      super(CNN3, self).__init__()
      
      # --- PARÁMETROS DE REGULARIZACIÓN ---
      # Aumentar la tasa de Dropout en FC para combatir el overfitting.
      DROPOUT_RATE_FC = 0.6  # Incrementado de 0.5 a 0.6
      
      # --- BLOQUE CONVOLUCIONAL 1 (32 -> 32) ---
      # Dos Conv 3x3 seguidas, manteniendo las dimensiones espaciales con padding=1.
      self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
      self.bn1_1 = nn.BatchNorm2d(32)
      
      self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
      self.bn1_2 = nn.BatchNorm2d(32)
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 32x16x16
      # Dropout después del pooling
      self.dropout_conv1 = nn.Dropout2d(0.1) 


      # --- BLOQUE CONVOLUCIONAL 2 (32 -> 64) ---
      self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
      self.bn2_1 = nn.BatchNorm2d(64)
      
      self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
      self.bn2_2 = nn.BatchNorm2d(64)
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 64x8x8
      self.dropout_conv2 = nn.Dropout2d(0.2) # Dropout ligeramente aumentado


      # --- BLOQUE CONVOLUCIONAL 3 (64 -> 128) ---
      self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
      self.bn3_1 = nn.BatchNorm2d(128)
      
      self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
      self.bn3_2 = nn.BatchNorm2d(128)
      self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 128x4x4
      self.dropout_conv3 = nn.Dropout2d(0.25)


      # --- CAPAS LINEALES (FULLY CONNECTED) ---
      # El tamaño de entrada ha cambiado debido al uso de padding=1, 
      # resultando en 4x4 al final (no 2x2 como tu modelo anterior)
      self.l1 = nn.Linear(in_features=128 * 4 * 4, out_features=512) # Aumento de características
      self.bn_fc1 = nn.BatchNorm1d(512)
      self.dropout_fc = nn.Dropout(DROPOUT_RATE_FC) 
      
      self.l2 = nn.Linear(in_features=512, out_features=10) # Salida Final

  def forward(self, x):
      # Bloque 1
      out = F.relu(self.bn1_1(self.conv1_1(x)))
      out = F.relu(self.bn1_2(self.conv1_2(out))) # Segunda Conv antes de Pool
      out = self.pool1(out)
      out = self.dropout_conv1(out)
      
      # Bloque 2
      out = F.relu(self.bn2_1(self.conv2_1(out)))
      out = F.relu(self.bn2_2(self.conv2_2(out)))
      out = self.pool2(out)
      out = self.dropout_conv2(out)
      
      # Bloque 3
      out = F.relu(self.bn3_1(self.conv3_1(out)))
      out = F.relu(self.bn3_2(self.conv3_2(out)))
      out = self.pool3(out)
      out = self.dropout_conv3(out)
      
      # Aplanar (Flatten)
      out = torch.flatten(out, 1) # Salida: (Batch, 128 * 4 * 4 = 2048)

      # Capas Denses
      out = F.relu(self.bn_fc1(self.l1(out)))
      out = self.dropout_fc(out) 
      
      out = self.l2(out)
      return out
  
class CNN4(nn.Module):
    def __init__(self, num_classes=10):
      super(CNN4, self).__init__()
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1)
      self.conv2 = nn.Conv2d(in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
      self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
      self.conv4 = nn.Conv2d(in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
      self.conv5 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
      self.conv6 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
      self.conv7 = nn.Conv2d(in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
      self.conv8 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
      self.conv9 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)

      self.bn1 = nn.BatchNorm2d(32 * 2)
      self.bn2 = nn.BatchNorm2d(128 * 2)
      self.bn3 = nn.BatchNorm2d(256 * 2)

      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.dropout = nn.Dropout2d(0.2)

      self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
      self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
      self.fc3 = nn.Linear(2048 * 2, num_classes)
      self.relu = nn.ReLU()

    def forward(self, x):
      x = self.relu(self.bn1(self.conv1(x)))
      x = self.relu(self.conv2(x))
      x = self.relu(self.conv3(x))
      x = self.maxpool(x)

      x = self.relu(self.bn2(self.conv4(x)))
      x = self.relu(self.conv5(x))
      x = self.relu(self.conv6(x))
      x = self.maxpool(x)
      x = self.dropout(x)

      x = self.relu(self.bn3(self.conv7(x)))
      x = self.relu(self.conv8(x))
      x = self.relu(self.conv9(x))
      x = self.maxpool(x)
      x = self.dropout(x)

      x = torch.flatten(x, start_dim=1)
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.dropout(x)
      x = self.fc3(x)
      return x
