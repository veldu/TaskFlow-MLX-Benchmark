# TaskFlow: MLX Benchmark Laboratory

Este repositorio contiene el entorno de pruebas aislado para el estudio empírico de rendimiento de modelos de lenguaje (LLMs) locales, desarrollado como parte del Trabajo de Fin de Grado en Ingeniería Informática centrado en el proyecto **TaskFlow**.

## ¿Qué es TaskFlow?
TaskFlow es un concepto de aplicación de escritorio de "Productividad Generativa" de arquitectura híbrida (Electron, React, FastAPI). A diferencia de los asistentes conversacionales estándar (chatbots), TaskFlow actúa como un orquestador estructural: el usuario describe una necesidad organizativa (ej. "Planificar el desarrollo de mi proyecto") y el sistema de IA local analiza la petición para generar y renderizar dinámicamente una interfaz gráfica interactiva (Widgets como tableros Kanban, sistemas de notas o calculadoras académicas) pre-rellenada con datos lógicos. Todo el procesamiento ocurre de manera 100% local, garantizando coste cero de inferencia y privacidad absoluta.

## Objetivo del Estudio
Validar empíricamente la viabilidad de ejecutar este motor de generación de interfaces en hardware de consumo utilizando el framework **Apple MLX**. El laboratorio evalúa tres pilares críticos para definir la arquitectura final del sistema (Advanced Model Cascading):

1. **Latencia:** *Time To First Token* (TTFT) y *Tokens Per Second* (TPS) para garantizar una experiencia de usuario responsiva (*Bias for Action*).
2. **Eficiencia:** Consumo de Memoria Unificada (VRAM) y carga de computación en GPU.
3. **Fiabilidad:** Capacidad de seguimiento de instrucciones complejas en contextos restrictivos y generación de esquemas JSON estrictamente válidos para la UI (Pydantic y TypeScript Injection).

> *Nota: Para una lectura detallada sobre las conclusiones arquitectónicas derivadas de estas pruebas, consulta el archivo `results_and_conclusions.md`.*

## Configuración del Entorno
El estudio se realiza en un entorno virtual aislado para evitar conflictos de dependencias y garantizar la pureza de las métricas obtenidas.

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias (MLX, MLX-LM, Pydantic, Pandas)
pip install -r requirements.txt
```

## Ejecución de las Pruebas

El laboratorio cuenta con dos orquestadores principales para ejecutar las baterías de pruebas. Es imprescindible tener el entorno virtual activo antes de ejecutarlos.

### 1. Suite de Benchmarks Interactiva
Permite seleccionar dinámicamente el modelo a evaluar y la prueba específica que se desea correr. Evalúa distintos componentes de la arquitectura:
* **Generación de UI (Prompt Injection):** Valida la capacidad del modelo para generar estructuras de datos exactas comparando técnicas como JSON Schema nativo vs. Interfaces TypeScript.
* **Enrutamiento Lógico (Router / Intent Classification):** Evalúa la capacidad de modelos ultraligeros (< 4B) para realizar triaje de intenciones y selección de herramientas.
* **Test de Estrés E2E (Context Bloat):** Simula cargas de base de datos de hasta 17.000 tokens para identificar los límites cognitivos (*Lost in the Middle*) y justificar la necesidad de un pipeline RAG.

```bash
python run_benchmark_suite.py
```

### 2. Benchmark de Robustez y Varianza
Script de automatización diseñado para ejecutar pruebas empíricas iterativas. Permite seleccionar un modelo y someterlo a múltiples ejecuciones consecutivas (N iteraciones) del mismo test para calcular medias estadísticas fiables, evaluar la degradación del rendimiento con el calentamiento térmico y medir la consistencia del *output*.

```bash
python run_robust_benchmark.py
```

## Estructura del Proyecto
* `/datasets`: Colecciones de *prompts* de prueba, casos de uso complejos y volcados simulados de bases de datos para estresar el contexto.
* `/results`: Directorio de salida donde se almacenan los *logs* de ejecución y los archivos `.csv` con la telemetría recolectada.
* `/scripts`: Lógica central de medición, funciones base de MLX y módulos de validación de esquemas Pydantic.

## Especificaciones de Hardware y Advertencias de Reproducibilidad

Los resultados base de este laboratorio han sido obtenidos utilizando la siguiente configuración de hardware:
* **Modelo:** MacBook Pro (2024)
* **Chip:** Apple M4 Pro (12-core CPU, 16-core GPU)
* **Memoria:** 24 GB Unified Memory Architecture (UMA)
* **Ancho de banda de memoria:** 273 GB/s

**Nota Importante sobre Memoria Unificada (UMA):**
Los resultados de latencia, éxito estructural y, sobre todo, viabilidad de ejecución, variarán drásticamente dependiendo del hardware del sistema host. Apple Silicon utiliza memoria unificada, lo que significa que la CPU y la GPU comparten el mismo *pool* de memoria RAM. 

* Si se intenta replicar estos tests en máquinas con **8 GB o 16 GB de memoria**, la carga de modelos de 14B parámetros (que requieren aproximadamente 9.5 GB de VRAM en cuantización a 4-bit, más la *KV Cache* de contextos largos) resultará en cuellos de botella severos, uso excesivo del archivo de paginación (SWAP) en el SSD, o errores directos de *Out of Memory* (OOM). 
* Para perfilar modelos de gran tamaño en equipos de 16 GB o inferiores, es imperativo cerrar aplicaciones secundarias pesadas (navegadores web, contenedores Docker) para liberar la mayor cantidad de Memoria Unificada posible antes de lanzar los scripts.