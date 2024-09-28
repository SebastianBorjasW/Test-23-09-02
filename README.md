# Test-23-09-02

### Descripción:

Se le solicita desarrollar un programa en Python que clasifique imágenes en al menos tres categorías diferentes utilizando técnicas de Transfer Learning con modelos pre-entrenados de Deep Learning. El objetivo es construir un modelo que pueda reconocer y clasificar nuevas imágenes con una precisión razonable.

### Requisitos:

#### Selección del Conjunto de Datos:

Puede utilizar conjuntos de datos públicos como:
Kaggle Datasets: Animals-10 Dataset (contiene imágenes de 10 tipos de animales).
Conjunto de Datos Personalizado: Puede recopilar imágenes de tres categorías a su elección (por ejemplo, gatos, perros y pájaros).
Asegúrese de tener un conjunto equilibrado de imágenes para cada categoría.

#### Preprocesamiento de Datos:

Organizar las imágenes en directorios separados por clase para facilitar la carga y etiquetado.
Aplicar técnicas de aumento de datos (data augmentation) como rotación, volteo y escalado para aumentar la variedad del conjunto de datos.
Redimensionar las imágenes a un tamaño adecuado para el modelo pre-entrenado seleccionado (por ejemplo, 224x224 píxeles).
Implementación del Modelo:

#### Utilizar un modelo pre-entrenado disponible en Keras o PyTorch:

Congelar las capas del modelo pre-entrenado y agregar capas densas adicionales para adaptarlo a su problema de clasificación.
Compilar el modelo con una función de pérdida adecuada (por ejemplo, categorical_crossentropy) y un optimizador como Adam.
Entrenamiento y Validación:

Dividir el conjunto de datos en conjuntos de entrenamiento y validación (por ejemplo, 80% entrenamiento y 20% validación).
Entrenar el modelo y monitorear la precisión y la pérdida en ambos conjuntos.
Ajustar hiperparámetros como la tasa de aprendizaje y el número de épocas para mejorar el rendimiento.

#### Evaluación del Modelo:

Evaluar el modelo utilizando métricas como:
Precisión (Accuracy)
Generar gráficos que muestren:
La curva de precisión y pérdida durante el entrenamiento.
La matriz de confusión para visualizar el rendimiento por clase.

#### Prueba con Nuevas Imágenes:

Probar el modelo con imágenes nuevas que no formen parte del conjunto de datos utilizado.
Mostrar las predicciones y la probabilidad asociada a cada clase.

### Entregable

Grabar un video (máximo 5 minutos) mostrando:
La explicación del enfoque y las decisiones tomadas.
El código fuente y su estructura.
La ejecución del programa y demostración de su funcionamiento con diferentes imágenes.
Los resultados obtenidos y conclusiones.

### Entrega:

Código fuente bien organizado y documentado.
Archivo README con instrucciones para ejecutar el programa y explicaciones adicionales si es necesario.
El video de demostración en un formato común (por ejemplo, MP4, AVI).

Asegúrese de que el código sea reproducible; incluya instrucciones claras sobre cómo instalar dependencias y ejecutar el programa.
Si utiliza notebooks de Jupyter, proporcione el archivo .ipynb y exporte una versión en HTML o PDF.

#### Consejos:

Comience por explorar y comprender el conjunto de datos elegido.
Verifique que las imágenes estén correctamente etiquetadas y organizadas.
Realice pruebas con una pequeña parte del conjunto de datos para agilizar el proceso de desarrollo.
Documente las decisiones tomadas y los desafíos encontrados durante el desarrollo.

# Instrucciones para correr el proyecto 

1. Clona el repositorio:

   ```bash
   git clone https://github.com/SebastianBorjasW/Test-23-09-02
   ```
   
2. Accede al directorio Test-23-09-02:

   ```bash
   cd Test-23-09-02
   ```

3. Accede al directorio `app`:

   ```bash
   cd app
   ```

4. Instala las dependencias de Node:

   ```bash
   npm install
   ```

5. Instala axios:

   ```bash
   npm install axios
   ```

6. Corre el frontend con el siguiente comando:

   ```bash
   npm run dev
   ```

7. Abre otra terminal en la ruta del proyecto y accede al directorio del backend:

   ```bash
   cd backend
   ```

8. Instala las dependencias de PyTorch:

   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 
   ```

9. Instala FastAPI:

   ```bash
   pip install fastapi
   ```

10. Instala Uvicorn:

   ```bash
   pip install "uvicorn[standard]"
   ```

11. Instala `python-multipart`:

   ```bash
   pip install python-multipart
   ```

12. Descarga el archivo del siguiente link:

   Link: [Enlace aquí]

13. Muévete a la carpeta `models`:

   ```bash
   cd Test-23-09-02/backend/models
   ```

14. Mueve el archivo descargado:

   Mueve el archivo descargado `model2.pth` dentro de la carpeta `models`. Esto se hace porque el archivo es suficientemente grande como para no poder subirse a GitHub.

15. Asegúrate de estar en la carpeta `backend` y corre la aplicación:

   ```bash
   uvicorn main:app --reload
   ```

   Si no se detecta `uvicorn`, intenta con el siguiente comando: 

   ```bash
   python -m uvicorn main:app --reload
   ```
