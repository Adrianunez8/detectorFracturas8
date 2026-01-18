import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import io

# Configuración de la página
st.set_page_config(
    page_title="Segmentador de Fracturas de Humero",
    page_icon="🦴",
    layout="wide"
)

# Cargar los logos
logo_unl = Image.open("logoUNL.png")
logo_carrera = Image.open("LogoCarreraNombre.png")

# Crear columnas para los logos en la esquina superior derecha
col1, col2, col3 = st.columns([0.4, 0.2, 0.2])

with col1:
    st.title("Segmentador de Fracturas de Húmero")

with col2:
    st.image(logo_unl, use_container_width=True)

with col3:
    st.image(logo_carrera, use_container_width=True)

st.markdown("---")

# ============ NUEVO: FUNCIONES PARA CLASIFICACIÓN ============

# Definir la arquitectura del modelo de clasificación - MobileNetV3-Small
def crear_modelo_clasificador():
    """
    Crea el modelo MobileNetV3-Small con clasificador personalizado
    Debe coincidir con la arquitectura usada en el entrenamiento
    """
    try:
        from torchvision import models
        
        # Cargar MobileNetV3-Small (sin pesos pre-entrenados para la inferencia)
        try:
            # Intentar con la nueva API
            modelo = models.mobilenet_v3_small(weights=None)
        except:
            # Fallback para versiones antiguas
            modelo = models.mobilenet_v3_small(pretrained=False)
        
        # Modificar clasificador (igual que en el entrenamiento)
        modelo.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
        return modelo
        
    except Exception as e:
        st.error(f"Error creando arquitectura MobileNetV3: {e}")
        return None

@st.cache_resource
def load_classification_model():
    """
    Carga el modelo de clasificación PyTorch (MobileNetV3-Small)
    """
    try:
        from torchvision import models
        
        model_path = "mobilenet_mejor.pth"  # Ajusta el nombre de tu archivo
        
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Clasificador no encontrado: {model_path}")
            st.info("La aplicación continuará sin validación de radiografía")
            return None
        
        # Crear la arquitectura MobileNetV3-Small
        model = crear_modelo_clasificador()
        
        if model is None:
            return None
        
        # Cargar los pesos
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Si guardaste solo state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()  # Modo evaluación
        #Clasificador MobileNetV3-Small cargado
        return model
        
    except Exception as e:
        st.warning(f"No se pudo cargar el clasificador: {e}")
        st.info("La aplicación continuará sin validación de radiografía")
        return None

def preprocess_for_classification(image, img_size=(256, 256)):
    """
    Preprocesa la imagen para el clasificador PyTorch
    """
    try:
        # Convertir a PIL si es necesario
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Convertir a RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transformaciones para PyTorch
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)  # Añadir dimensión batch
        return img_tensor
        
    except Exception as e:
        st.error(f"Error en preprocesamiento para clasificación: {e}")
        return None

def classify_image(model, image, threshold=0.7):
    """
    Clasifica si la imagen es una radiografía
    Retorna: (es_radiografia, probabilidad)
    """
    try:
        # Preprocesar imagen
        img_tensor = preprocess_for_classification(image)
        
        if img_tensor is None:
            return False, 0.0
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prob_radiografia = probabilities[0][1].item()  # Probabilidad de ser radiografía
            
        es_radiografia = prob_radiografia >= threshold
        
        return es_radiografia, prob_radiografia
        
    except Exception as e:
        st.error(f"Error en clasificación: {e}")
        return False, 0.0

# ============ FUNCIONES ORIGINALES DE SEGMENTACIÓN ============

@st.cache_resource
def load_default_model():
    try:
        # Ruta fija donde tienes tu modelo
        model_path = "unet_fracturas_aug_opencv_fp16.h5"  # o la ruta donde está tu modelo
        
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            return model
        else:
            st.error(f"No se encontró el modelo en: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error al cargar el modelo por defecto: {e}")
        return None

def preprocess_image(image, img_size=(256, 256)):
    """
    Preprocesa la imagen para que sea compatible con el modelo U-Net
    """
    try:
        # Convertir PIL Image a array numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            # Si es un archivo subido
            image = Image.open(image)
            img_array = np.array(image)
        
        # Si la imagen es RGBA, convertir a RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        # Si es escala de grises, convertir a RGB
        elif len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Redimensionar a 256x256 (tamaño que espera el modelo)
        img_resized = cv2.resize(img_array, img_size)
        
        # Normalizar
        img_input = img_resized.astype(np.float32) / 255.0
        
        return img_input, img_resized
    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        return None, None

def predict_and_visualize(model, image_path_or_array, threshold=0.5):
    """
    Realiza la predicción y crea la visualización con umbral ajustable
    """
    try:
        # Preprocesar imagen
        img_input, img_resized = preprocess_image(image_path_or_array, img_size=(256, 256))
        
        if img_input is None:
            return None, None, None, None
        
        # Realizar predicción
        pred = model.predict(np.expand_dims(img_input, axis=0))
        
        # Extraer la máscara de predicción
        if len(pred.shape) == 4:
            pred_mask = pred[0, ..., 0]  # Tomar primer canal si hay múltiples
        else:
            pred_mask = pred[0, ...]  # Si ya es 2D
        
        # Aplicar el umbral seleccionado
        pred_bin = (pred_mask > threshold).astype(np.uint8)
        
        # Crear overlay
        alpha = 0.5  # transparencia
        overlay_pred = img_resized.copy()
        
        # Asegurarse de que las dimensiones coincidan
        if pred_bin.shape != overlay_pred.shape[:2]:
            pred_bin = cv2.resize(pred_bin, (overlay_pred.shape[1], overlay_pred.shape[0]))
        
        # Crear máscara roja
        red_mask = np.zeros_like(overlay_pred)
        red_mask[..., 0] = 255  # Canal rojo
        
        # Aplicar la máscara donde pred_bin > 0
        for c in range(3):
            overlay_pred[..., c] = np.where(
                pred_bin > 0, 
                red_mask[..., c] * alpha + overlay_pred[..., c] * (1 - alpha), 
                overlay_pred[..., c]
            )
        
        # Convertir a uint8 para visualización
        overlay_pred = overlay_pred.astype(np.uint8)
        
        return img_resized, overlay_pred, pred_bin, pred_mask
        
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None, None, None, None

def create_comparison_figure(original, prediction, threshold):
    """
    Crea una figura con comparación lado a lado
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Imagen original
    ax1.imshow(original)
    ax1.set_title("Imagen Original", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Imagen con predicción
    ax2.imshow(prediction)
    ax2.set_title(f"Predicción U-Net (Umbral: {threshold})", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def create_probability_heatmap(original, prob_map, threshold):
    """
    Crea un heatmap de las probabilidades
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    ax1.imshow(original)
    ax1.set_title("Imagen Original", fontsize=12)
    ax1.axis('off')
    
    # Heatmap de probabilidades
    im = ax2.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    ax2.set_title("Mapa de Probabilidades", fontsize=12)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Línea de umbral
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1)
    ax2.text(0.5, -0.1, f'Umbral: {threshold}', transform=ax2.transAxes, 
             ha='center', va='top', color='white', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
    
    # Máscara binaria
    binary_mask = (prob_map > threshold).astype(np.uint8)
    ax3.imshow(binary_mask, cmap='gray')
    ax3.set_title(f"Máscara Binaria (>{threshold})", fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

# ============ SIDEBAR ============

# Control deslizante para el umbral
threshold = st.sidebar.slider(
    "Umbral de Segmentación",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Umbral más alto = menos sensibilidad, menos falsos positivos. Umbral más bajo = más sensibilidad, más falsos positivos."
)

# Mostrar explicación del umbral
st.sidebar.markdown("---")
st.sidebar.subheader("Guía del Umbral")

if threshold >= 0.7:
    st.sidebar.info("**Umbral ALTO (≥0.7)**\n- Menos falsos positivos\n- Puede perder fracturas sutiles\n- Bueno para confirmación")
elif threshold <= 0.3:
    st.sidebar.warning("**Umbral BAJO (≤0.3)**\n- Máxima sensibilidad\n- Más falsos positivos\n- Bueno para screening")
else:
    st.sidebar.success("**Umbral MEDIO (0.4-0.6)**\n- Balance sensibilidad/especificidad\n- Bueno para uso general")

# Cargar los modelos
model = load_default_model()
classifier_model = load_classification_model()  # NUEVO

# Mostrar información del modelo en el sidebar
with st.sidebar:
    st.markdown("---")
    if model is not None:
        st.success("Modelo U-Net Listo")
        
        # Mostrar información del modelo
        st.subheader("Información del Modelo")
        st.write(f"**Entradas:** {model.input_shape}")
        st.write(f"**Salidas:** {model.output_shape}")
        st.write(f"**Capas:** {len(model.layers)}") 
        st.write(f"**Parámetros:** {model.count_params():,}")
    else:
        st.error("❌ No se pudo cargar el modelo. La aplicación no puede continuar.")
        st.stop()  # Detener la ejecución si no hay modelo

# ============ ÁREA PRINCIPAL ============

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Subir Imagen Médica")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen radiográfica",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formatos soportados: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        # Información de la imagen
        st.info("**Información de la imagen:**")
        st.write(f"**Tamaño original:** {image.size} pixels")

with col2:
    st.subheader("Resultados de la Segmentación")
    
    if uploaded_file is not None and model is not None:
        
        # NUEVO: PASO 1 - CLASIFICACIÓN (si el clasificador está disponible)
        puede_segmentar = True
        
        if classifier_model is not None:
            st.markdown("Validación de Radiografía")
            
            with st.spinner("Verificando si es una radiografía..."):
                es_radiografia, prob_radiografia = classify_image(
                    classifier_model, 
                    uploaded_file, 
                    threshold=0.7 #nivel de clasificacion de la imagen 
                )
            
            # Mostrar resultado de clasificación
                if es_radiografia:
                    st.success(f"**ES RADIOGRAFÍA**")
                else:
                    st.error(f"**NO ES RADIOGRAFÍA**")
                    puede_segmentar = False
            
            # Barra de progreso visual
            st.markdown("---")
        
        # PASO 2 - SEGMENTACIÓN (solo si pasó la clasificación o no hay clasificador)
        if puede_segmentar:
            # Realizar predicción con el umbral seleccionado
            with st.spinner("Realizando segmentación U-Net..."):
                original_img, overlay_img, mask, prob_map = predict_and_visualize(
                    model, uploaded_file, threshold=threshold
                )
            
            if original_img is not None and overlay_img is not None:
                # Mostrar resultados
                st.success(f"Segmentación completada (Umbral: {threshold})")
                
                # Crear y mostrar figura de comparación
                fig = create_comparison_figure(original_img, overlay_img, threshold)
                st.pyplot(fig)
                
                # Mostrar heatmap de probabilidades
                with st.expander("Ver Mapa de Probabilidades Detallado"):
                    if prob_map is not None:
                        heatmap_fig = create_probability_heatmap(original_img, prob_map, threshold)
                        st.pyplot(heatmap_fig)
                
                # Métricas de la segmentación
                st.subheader("Métricas de Segmentación")
                
                if mask is not None:
                    total_pixels = mask.shape[0] * mask.shape[1]
                    fracture_pixels = np.sum(mask)
                    fracture_percentage = (fracture_pixels / total_pixels) * 1000
                    
                    # Calcular estadísticas de probabilidad
                    if prob_map is not None:
                        mean_prob = np.mean(prob_map)
                        max_prob = np.max(prob_map)
                        prob_above_threshold = np.mean(prob_map > threshold) * 100
                    
                    col1_metric, col2_metric, col3_metric = st.columns(3)
                    
                    with col1_metric:
                        st.metric(
                            label="Área de fractura",
                            value=f"{fracture_percentage:.2f}%",
                            help="Porcentaje del área identificada como fractura"
                        )
                    
                    with col2_metric:
                        st.metric(
                            label="Píxeles detectados",
                            value=f"{fracture_pixels:,}",
                            help="Número total de píxeles sobre el umbral"
                        )
                    
                    with col3_metric:
                        if prob_map is not None:
                            st.metric(
                                label="Probabilidad media",
                                value=f"{mean_prob:.3f}",
                                help="Probabilidad promedio en toda la imagen"
                            )
                
                # Interpretación clínica dinámica
                st.subheader("Interpretación Clínica")
                
                if fracture_percentage > 5:
                    st.error("""
                    **🩺 ALTA fractura extensa**
                    - Se observan áreas significativas de fractura
                    - Urgencia: Alta
                    - Se recomienda evaluacion médica inmediata
                    """)
                elif fracture_percentage > 3:
                    st.warning("""
                    **🩺Fractura media**
                    - Se detectaron áreas de posible fractura
                    - Urgencia: Media
                    - Se recomienda evaluación médica especializada
                    """)
                elif fracture_percentage > 0.8:
                    st.warning("""
                    **🩺 Fractura menor**
                    - Pequeñas áreas sospechosas detectadas
                    - Urgencia: Baja
                    - Se sugiere evaluación médica adicional
                    """)
                else:
                    st.success("""
                    **Sin indicios aparentes de fractura**
                    - No se detectaron áreas significativas de fractura
                    - Urgencia: Ninguna
                    - Continuar con controles habituales
                    """)
                    
                # Advertencia sobre el umbral
                if threshold > 0.7 and fracture_percentage < 1:
                    st.info("**Nota:** Con un umbral tan alto, es posible que se pasen por alto fracturas sutiles. Considera reducir el umbral para screening.")
                elif threshold < 0.3 and fracture_percentage > 5:
                    st.info("**Nota:** Con un umbral tan bajo, pueden incluirse áreas normales. Considera aumentar el umbral para confirmación.")
        
        else:
            # NUEVO: Mensaje cuando NO es radiografía
            st.warning(f"""
            La imagen NO es una radiografía.
            
            El sistema ha determinado que la imagen cargada no corresponde a una radiografía médica.
            
            **Por favor:**
            - Verifica que hayas seleccionado la imagen correcta.
            - Asegúrate de que sea una radiografía de calidad diagnóstica.
            - Intenta con otra imagen.
            """)
    
    elif uploaded_file is not None and model is None:
        st.warning("""
        **Modelo U-Net no cargado**
        Por favor selecciona y carga tu modelo .h5 en la barra lateral.
        """)
    else:
        st.info("""
        **📋 Instrucciones:**
        1. Selecciona una imagen radiográfica desde tu PC
        2. El sistema verificará automáticamente si es una radiografía válida
        3. Si es válida, se realizará la segmentación de fracturas
        4. Los resultados aparecerán con métricas detalladas
        
        **Nota:** Solo se procesarán imágenes identificadas como radiografías
        """)

# Sección educativa sobre umbrales
st.markdown("---")
st.subheader("Guía de Umbrales de Segmentación")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    ### 🔴 Umbral Alto (0.7-0.9)
    **Ventajas:**
    - Muy pocos falsos positivos
    - Alta confianza en las detecciones
    - Ideal para confirmación
    
    **Desventajas:**
    - Puede perder fracturas sutiles
    - Menos sensibilidad
    """)

with col_info2:
    st.markdown("""
    ### 🟡 Umbral Medio (0.4-0.6)
    **Ventajas:**
    - Balance ideal
    - Buen rendimiento general
    - Uso estándar recomendado
    
    **Desventajas:**
    - Compromiso entre sensibilidad y especificidad
    """)

with col_info3:
    st.markdown("""
    ### 🟢 Umbral Bajo (0.1-0.3)
    **Ventajas:**
    - Máxima sensibilidad
    - Detecta fracturas muy sutiles
    - Ideal para screening
    
    **Desventajas:**
    - Muchos falsos positivos
    - Requiere más revisión manual
    """)

# Pie de página
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Sistema de Segmentación de Fracturas U-Net - Umbral ajustable<br>"
    "Los resultados deben ser interpretados por profesionales de la salud<br>"
    "Todos los derechos reservados | Copyright 2025"
    "</div>",
    unsafe_allow_html=True
)