import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchinfo import summary
from typing import Tuple
from io import StringIO
from contextlib import redirect_stdout
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
import multiprocessing

from models import *

# ** Helper Functions

def parse_arguments():
    """
    Define y parsea los argumentos de la línea de comandos para la configuración del entrenamiento.
    """
    parser = argparse.ArgumentParser(
        description="Script de entrenamiento para CNN en CIFAR-10 con configuración ajustable.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 1. Parámetros Numéricos/Simples
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Semilla (Seed) para asegurar la reproducibilidad. (Default: 42)'
    )
    parser.add_argument(
        '--lr', 
        type=int, 
        default=0.001, 
        help='Semilla (Seed) para asegurar la reproducibilidad. (Default: 42)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='Tamaño del lote (Batch Size) para los Data Loaders. (Default: 64)'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=50, 
        help='Número de épocas de entrenamiento. (Default: 50)'
    )

    # 2. Parámetros de String (Modelo y Dataset)
    parser.add_argument(
        '--model', 
        type=str, 
        default='CNN1', 
        # choices=['CNN1', 'CNN2'],
        help='Modelo a utilizar. (Default: CNN1). Opciones: CNN1, CNN_Regularized'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='CIFAR10', 
        choices=['CIFAR10', 'MNIST'],
        help='Dataset a cargar. Actualmente solo soporta CIFAR10. (Default: CIFAR10)'
    )

    # 3. Parámetros de Partición del Dataset (Dataset Partition)
    # Se requiere que sean 3 números separados por comas.
    parser.add_argument(
        '--partition', 
        type=str, 
        default='80,10,10', 
        help='Porcentaje de partición de datos (Train, Validation, Test) separados por comas. '
             'Debe sumar 100. (Default: 80,10,10)'
    )
    
    args = parser.parse_args()

    # --- Validación y Procesamiento de Partición (Lógica del 80,10,10) ---
    
    try:
        # Convertir la cadena '80,10,10' a una lista de enteros [80, 10, 10]
        ratios = [int(r.strip()) for r in args.partition.split(',')]
    except ValueError:
        print(f"Error: La partición debe ser una lista de números enteros separados por comas (ej. 80,10,10).")
        sys.exit(1)
        
    if len(ratios) != 3:
        print(f"Error: La partición debe tener exactamente 3 valores (Train, Validation, Test). Se encontraron {len(ratios)}.")
        sys.exit(1)
        
    if sum(ratios) != 100:
        print(f"Error: Los porcentajes de la partición deben sumar 100. Suma actual: {sum(ratios)}.")
        sys.exit(1)
        
    # Guardar la lista de enteros procesada en los argumentos
    args.dataset_partition = ratios
    
    print("\n--- Configuración de Ejecución ---")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"SEED: {args.seed}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Data split (T/V/T): {args.dataset_partition[0]}% / {args.dataset_partition[1]}% / {args.dataset_partition[2]}%")
    print("----------------------------------\n")
    
    return args

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, data_loader, criterion, device, arq):
    """
    Calcula la pérdida promedio, la precisión, y recopila las etiquetas
    verdaderas y predichas para métricas avanzadas.
    """
    model.eval() # Poner el modelo en modo evaluación
    total_loss = 0.0
    n_samples = 0

    # Listas para almacenar todas las etiquetas
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if arq.startswith("M"):
                images = images.view(images.size(0), -1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            
            # --- Almacenamiento de Etiquetas ---
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            
    avg_loss = total_loss / n_samples

    # Convertir a arrays de NumPy
    y_true = np.array(all_labels)
    y_pred = np.array(all_predicted)

    # Calcular Accuracy simple (para consistencia)
    accuracy = 100.0 * np.sum(y_true == y_pred) / n_samples

    model.train() # Devolver el modelo a modo entrenamiento

    # Retornar más métricas para el reporte avanzado
    return avg_loss, accuracy, y_true, y_pred

def save_learning_curves(history, path, filename):
    # 1. Definir los datos del eje X
    epochs = range(1, len(history['train_loss']) + 1)

    # 2. CREAR LA FIGURA Y LOS DOS SUBGRÁFICOS (1 fila, 2 columnas)
    # plt.figure(figsize=(12, 5)) crea la figura
    # ax1 y ax2 son las referencias a los dos subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # Aumenté un poco el tamaño

    # --- GRÁFICO DE PÉRDIDA (LOSS) en ax1 ---
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves (Pérdida vs. Época)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Pérdida)')
    ax1.legend()

    # --- GRÁFICO DE PRECISIÓN (ACCURACY) en ax2 ---
    # Convertimos la precisión a un rango de 0-100 para un gráfico más claro
    train_acc_pct = [acc * 1 for acc in history['train_acc']]
    val_acc_pct = [acc * 1 for acc in history['val_acc']]

    ax2.plot(epochs, train_acc_pct, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc_pct, 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy Curves (Precisión vs. Época)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (Precisión) (%)')
    ax2.legend()

    # 3. GUARDAR LA FIGURA COMPLETA UNA SOLA VEZ
    full_path = os.path.join(path, filename)
    plt.tight_layout() # Asegura que los títulos y etiquetas no se superpongan
    plt.savefig(full_path) 
    plt.close(fig) # Cierra la figura en memoria

    print(f"Learning curves saved to {full_path}")
    # Muestra el patrón de overfitting: la pérdida de validación sube mientras la de entrenamiento baja

def calculate_inference_speed(model, data_loader, device, arq):
      """Calcula el tiempo total de inferencia y las predicciones por segundo."""
      model.eval()
      total_images = 0
      start_time = time.time()
      
      with torch.no_grad():
          for images, labels in data_loader:
              images = images.to(device)
              # Solo hacemos el forward pass, no necesitamos las etiquetas para el tiempo
              if arq.startswith("M"):
                images = images.view(images.size(0), -1)
              _ = model(images)
              total_images += images.size(0)
      
      inference_time = time.time() - start_time
      # Evita la división por cero si el tiempo es cero
      predictions_per_second = total_images / inference_time if inference_time > 0 else 0 
      
      return inference_time, predictions_per_second

def save_confusion_matrix(test_y_true, test_y_pred, path, filename):
  cm_test = confusion_matrix(test_y_true, test_y_pred)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=True,
              xticklabels=CLASSES, yticklabels=CLASSES)
  plt.title('Confusion Matrix - Test Set')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
#   plt.show()
  # Save plot
  plt.tight_layout()
  plt.savefig(path + "/" + filename)
  plt.close()

def save_and_print_performance_metrics(metrics, path, filename):
  summary_content = ""
  summary_content += "==============================================\n"
  summary_content += "\tPerformance Metrics (Resumen)\n"
  summary_content += "==============================================\n"
  summary_content += f"1. Total Training Time:    {performance_metrics['total_training_time_s']:.2f} seconds\n"
  summary_content += f"2. Average Time per Epoch: {np.mean(metrics['epoch_times_s']):.2f} seconds\n"
  summary_content += f"3. Total Inference Time (Test Set): {performance_metrics['inference_time_s']:.2f} seconds\n"
  summary_content += f"4. Predictions per Second: {performance_metrics['predictions_per_second']:.2f} img/s\n"
  summary_content += "==============================================\n"

  print(summary_content)

  try:
    os.makedirs(path, exist_ok=True)
    FULL_PATH = os.path.join(path, filename)

    with open(FULL_PATH, 'w') as f:
        f.write(summary_content)
    print(f"Performance metrics saved sucessfully at: {FULL_PATH}")

  except Exception as e:
    print(f"Error while saving performance metrics at {FULL_PATH}: {e}")

def save_model_summary(
    model: nn.Module, 
    input_size: Tuple[int, int, int, int], 
    path,
    filename: str = 'model_summary.txt'
):
    """
    Genera el resumen de la arquitectura usando torchinfo y guarda la salida 
    en un archivo de texto para su fácil inclusión en LaTeX.
    
    Args:
        model: El modelo PyTorch instanciado (ej., CNN_Regularized()).
        input_size: La forma del tensor de entrada (Batch, Channels, H, W).
        path: Directorio donde guardar el archivo.
        filename: Nombre del archivo de salida (.txt).
    """
    
    os.makedirs(path, exist_ok=True)
    FULL_PATH = os.path.join(path, filename)

    try:
        # Abre el archivo para escritura
        with open(FULL_PATH, 'w', encoding='utf-8') as f:
            # 1. Redirecciona la salida estándar (stdout) al archivo
            sys.stdout = f
            
            print("=========================================================")
            print("          RESUMEN DE LA ARQUITECTURA (torchinfo)")
            print(f"      Input Size: {input_size}")
            print("=========================================================")

            # 2. Llama a torchinfo.summary
            summary(model, input_size=input_size, device='cpu') 
            
            # 3. Restaura la salida estándar a la consola
            sys.stdout = sys.__stdout__
        
        print(f"\nResumen del modelo guardado exitosamente en: {FULL_PATH}")
        
    except Exception as e:
        sys.stdout = sys.__stdout__ # Asegurarse de restaurar stdout en caso de error
        print(f"\nError al generar o guardar el resumen del modelo: {e}")

def save_classification_report_to_txt(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: list, 
    path: str = './results/reports', 
    filename: str = 'classification_report_final.txt'
):
    """
    Genera el reporte de clasificación de sklearn, lo imprime en consola
    y lo guarda en un archivo de texto plano.
    
    Args:
        y_true: Etiquetas verdaderas (NumPy array).
        y_pred: Etiquetas predichas (NumPy array).
        classes: Lista de nombres de las clases (ej., ('plane', 'car', ...)).
        path: Directorio donde guardar el archivo.
        filename: Nombre del archivo de salida.
    """
    
    os.makedirs(path, exist_ok=True)
    FULL_PATH = os.path.join(path, filename)

    # 1. Crear un buffer en memoria para capturar la salida impresa
    output_buffer = StringIO()
    
    # 2. Redirigir la salida estándar (print) al buffer
    with redirect_stdout(output_buffer):
        print("==================================================")
        print("       REPORTE DE CLASIFICACIÓN FINAL")
        print("==================================================")
        print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # 3. Obtener el contenido del buffer
    report_content = output_buffer.getvalue()

    # 4. Imprimir en consola (para que el usuario lo vea)
    print(report_content)

    # 5. Guardar en el archivo de texto usando UTF-8 para evitar errores de codificación
    try:
        # Usamos 'utf-8' para manejar cualquier carácter especial en los nombres de las clases
        with open(FULL_PATH, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nReporte de clasificación guardado exitosamente en: {FULL_PATH}")
        
    except Exception as e:
        print(f"Error al guardar el reporte en {FULL_PATH}: {e}")

m_path = ""
if __name__ == '__main__':
  config = parse_arguments()

  # Create directory to save results
  m_path = config.dataset + "_" + str(config.num_epochs) +  "epochs_" + config.model + "_" + config.partition + "_lr " + str(config.lr)
  try:
      os.makedirs(m_path, exist_ok=True)
      print(f"Directory '{m_path}' ensured to exist.")
  
  except OSError as e:
      print(f"Error creating directory: {e}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Device: ", device)
  
  # --- Definición de transformaciones de datos (Data augmentation) ---
  # --- Random crop y flip ---

  train_transform = transforms.Compose([
    # 4 píxeles de padding para el RandomCrop de 32x32
    transforms.RandomCrop(32, padding=4), 
    # Volteo aleatorio horizontal
    transforms.RandomHorizontalFlip(),
    # Convertir a tensor
    transforms.ToTensor(),
    # Normalización
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
  ])

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
  ])

  # --- Lectura del dataset ---
  train_original = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transforms.ToTensor(), download=True)
  # test_original = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
  # full_dataset = ConcatDataset([train_original, test_original])

  # --- Partición del dataset ---

  dataset_len = len(train_original)

  TRAIN_LEN = config.dataset_partition[0] / 100
  VALIDATION_LEN = config.dataset_partition[1] / 100
  TEST_LEN = config.dataset_partition[2] / 100

  TRAIN_SIZE = int(dataset_len * TRAIN_LEN)
  VALIDATION_SIZE = int(dataset_len * VALIDATION_LEN)
  TEST_SIZE = int(dataset_len * TEST_LEN)

  train_set, validation_set, test_set = random_split(
    train_original, 
    [TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE],
    generator=torch.Generator().manual_seed(config.seed)
  )

  print(f"Dataset Split: Train={len(train_set)}, Val={len(validation_set)}, Test={len(test_set)}")

  train_set.dataset.transform = train_transform
  validation_set.dataset.transform = test_transform
  test_set.dataset.transform = test_transform

  # --- Data Loaders (Para usar batch size) ---
  num_workers = multiprocessing.cpu_count()-1
  train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
  validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
  test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)

  models = {
      "MLP1" : MLP1,
      "CNN1" : CNN1,
    #   "CNN2" : CNN2,
      "CNN3" : CNN3,
      "CNN4" : CNN4,
  }

  model_str = config.model
  MODEL = models[model_str]
  
  model = MODEL().to(device)
  if model_str.startswith("M"):
    save_model_summary(model, (1,3072), m_path)
  else:
    save_model_summary(model, (1,3,32,32), m_path)
  num_params = count_trainable_parameters(model)
  print(f"Total Trainable Parameters: {num_params:,}")

  # -- Pérdida y Optimizador --
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0.0005)

  # -- Estructuras de datos para resultados --
  history = {
      'train_loss': [],
      'val_loss': [],
      'train_acc': [],
      'val_acc': []
  }

  performance_metrics = {
      'total_training_time_s': 0.0,
      'epoch_times_s': [],
      'inference_time_s': 0.0,
      'predictions_per_second': 0.0,
  }

  # -- Definir los nombres de las 10 clases de CIFAR-10 para el reporte --
  CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # -- Entrenamiento --
  print("\nStarting Training...")
  total_training_time_start = time.time()

  # Inicialización para el seguimiento del mejor modelo
  best_val_loss = float('inf')
  # Nombre de archivo para guardar los mejores pesos
  BEST_WEIGHTS_PATH = f'{m_path}/best_cnn_weights_{device}.pth'

  for epoch in range(1, config.num_epochs + 1):
      epoch_start_time = time.time()

      model.train()
      total_train_loss = 0.0
      n_train_samples = 0
      
      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device)
          labels = labels.to(device)

          # Forward, Backward, and Optimize
          if model_str.startswith("M"):
              images = images.view(images.size(0), -1)
          outputs = model(images)
          loss = criterion(outputs, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          total_train_loss += loss.item() * images.size(0)
          n_train_samples += images.size(0)

      # Cálculo y Almacenamiento del Tiempo por Época
      epoch_duration = time.time() - epoch_start_time
      performance_metrics['epoch_times_s'].append(epoch_duration) # 2. Guardar el tiempo por época
          
      # Calcular la pérdida de entrenamiento promedio de la época
      avg_train_loss = total_train_loss / n_train_samples
      history['train_loss'].append(avg_train_loss)
      
      # VALIDACIÓN
      # La precisión de entrenamiento necesita ser calculada fuera del bucle de optimización
      train_loss, train_acc, _, _ = evaluate_model(model, train_loader, criterion, device, model_str)
      val_loss, val_acc, val_y_true, val_y_pred = evaluate_model(model, validation_loader, criterion, device, model_str)
      
      # Almacenar métricas
      history['train_acc'].append(train_acc)
      history['val_loss'].append(val_loss)
      history['val_acc'].append(val_acc)
      
      print(f'--- Epoch {epoch} Final --- '
            f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% | '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
      
      # GUARDAR LOS MEJORES PESOS
      if val_loss < best_val_loss:
          print(f"[SAVING BEST MODEL] New best Val Loss: {val_loss:.4f} (was {best_val_loss:.4f}). Saving weights to {BEST_WEIGHTS_PATH}")
          best_val_loss = val_loss
          # Guardar solo los pesos del modelo (state_dict)
          torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
      
      # REPORTE DE CLASIFICACIÓN CADA 10 ÉPOCAS
      if epoch % 10 == 0:
          print("\n[VALIDATION REPORT]")
          # Imprimir el reporte detallado
          print(classification_report(val_y_true, val_y_pred, target_names=CLASSES, zero_division=0))
          # Imprimir la matriz de confusión
          print("Confusion Matrix:\n", confusion_matrix(val_y_true, val_y_pred))

  # --- CÁLCULO DEL TIEMPO TOTAL DE ENTRENAMIENTO ---
  total_training_duration = time.time() - total_training_time_start
  performance_metrics['total_training_time_s'] = total_training_duration # 1. Guardar el tiempo total

  # --- GRAFICAR CURVAS DE APRENDIZAJE ---
  save_learning_curves(history, m_path, "learning_curves.png")

  # --- EVALUACIÓN FINAL EN TEST SET ---
  print('\nTraining Finished. Evaluating on Test Set...')

  # GET SAVED MODEL (BEST ONE)
  if os.path.exists(BEST_WEIGHTS_PATH):
    print(f"Loading best weights from {BEST_WEIGHTS_PATH} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH))

  # CÁLCULO DEL TIEMPO DE INFERENCIA (PREDICTIONS PER SECOND) EN EL CONJUNTO DE PRUEBA ---

  inference_time, predictions_per_second = calculate_inference_speed(model, test_loader, device, model_str)
  performance_metrics['inference_time_s'] = inference_time 
  performance_metrics['predictions_per_second'] = predictions_per_second

  # Capturar las etiquetas verdaderas y predichas del conjunto de prueba
  test_loss, test_acc, test_y_true, test_y_pred = evaluate_model(model, test_loader, criterion, device, model_str)

  print(f'\n[FINAL RESULTS] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

  # REPORTE FINAL MEDIDAS DE DESEMPEÑO
  print("\n[FINAL TEST CLASSIFICATION REPORT]")
  save_classification_report_to_txt(test_y_true, test_y_pred, CLASSES, m_path, "Final classification report (test set).txt")

  print("\nFINAL TEST CONFUSION MATRIX (True Labels vs. Predicted Labels):")
  print(confusion_matrix(test_y_true, test_y_pred))

  # MATRIZ DE CONFUSIÓN
  print("\nFINAL TEST CONFUSION MATRIX (True Labels vs. Predicted Labels):")
  save_confusion_matrix(test_y_true, test_y_pred, m_path, "confusion_matrix.png")

  # REPORTE FINAL DE TIEMPOS
  save_and_print_performance_metrics(performance_metrics, m_path, "performance_metrics_test.txt")