# TaskFlow: MLX Benchmark Laboratory

Este repositorio contiene el entorno de pruebas aislado para el estudio de rendimiento de modelos de lenguaje (LLMs) locales en el proyecto **TaskFlow** (TFG Ingeniería Informática).

##  Objetivo del Estudio
Validar empíricamente el rendimiento de diferentes modelos cuantizados bajo el framework **Apple MLX**, optimizando la ejecución para el chip **M4 Pro**. Se evalúan tres pilares críticos:
1. **Latencia:** Time To First Token (TTFT) y Tokens Per Second (TPS).
2. **Eficiencia:** Consumo de memoria unificada (VRAM) y carga de CPU/GPU.
3. **Fiabilidad:** Capacidad de seguimiento de instrucciones (Instruction Following) y generación de esquemas JSON válidos para la UI Generativa.

## Configuración del Entorno
El estudio se realiza en un entorno aislado para garantizar la pureza de las métricas.

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Estructura del Proyecto
- `/datasets`: Prompts de prueba diseñados para estresar la lógica de Generative UI.
- `/results`: Logs y archivos CSV con las métricas obtenidas.
- `/scripts`: Lógica de medición y benchmarking.
- `run_tests.py`: Orquestador principal de las pruebas.

## Especificaciones del Hardware de Prueba
- **Modelo:** MacBook Pro (2024)
- **Chip:** Apple M4 Pro (12-core CPU, 16-core GPU)
- **Memoria:** 24 GB Unified Memory
