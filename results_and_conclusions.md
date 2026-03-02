# Documento de Conclusiones sobre los Modelos de Inferencia para TaskFlow

**Fecha:** Marzo 2026  
**Entorno:** Apple MacBook Pro M4 Pro (24GB UMA) | MLX Framework

### Objetivos Iniciales
1. Estudio de la **viabilidad** de aplicar modelos de inferencia local a la aplicación de optimización de flujos de trabajo TaskFlow.
2. Estudio de técnicas de ingeniería de prompts e inyección de esquemas para un rendimiento óptimo y tiempos de espera interactivos para el usuario.
3. Comparativa de distintos modelos, cuantizaciones y número de parámetros, evaluando sus tiempos de respuesta, fiabilidad estructural y rendimiento cognitivo.

---

## 1. Registro de Benchmark: Evaluación de Inyección de Esquemas (Generative UI)

> **Modelo principal de control:** `Qwen2.5-14B-Instruct-4bit`.  
> Se seleccionó este modelo por ser lo suficientemente robusto para observar claramente el efecto de las diferentes técnicas de prompting y cómo afectan a la velocidad de generación (*Prefill* y *Decoding*) sin sufrir caídas graves de razonamiento.

### Objetivo
Evaluar la viabilidad de generar estructuras JSON estrictas y anidadas para la UI de TaskFlow, buscando el equilibrio óptimo entre **Rendimiento** (Latencia), **Fiabilidad Estructural** (Pydantic) y **Mantenibilidad** (*Clean Code*).

### Resultados Empíricos (Media Generación UI)

| Estrategia / Test | Éxito JSON | Input Tokens | Latencia Media (TTFT) | Velocidad (TPS) | VRAM Pico | Mantenibilidad |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (Hardcoded)** | 100% | ~300 | ~1.4s | 26.0 | ~8.3 GB | Muy Baja |
| **2. JSON Schema OpenAPI** | 100% | ~1111 | ~5.4s | 20.3 | ~8.7 GB | Muy Alta |
| **3. Esquema Minificado** | 100% | ~450 | ~2.2s | 24.8 | ~8.4 GB | Muy Alta |
| **4. Interfaces TypeScript** | 100% | ~430 | ~2.2s | 25.0 | ~8.4 GB | Muy Alta |

### Análisis por Iteraciones

* **Iteración 1: Plantilla Manual (Hardcoded Prompt)**
  * **Técnica:** Se inyecta la estructura JSON completa y vacía como *string* en el prompt del sistema.
  * **Problema:** Anti-patrón de diseño. Genera un acoplamiento fuerte y viola el principio DRY. Si el frontend (React) cambia una interfaz, hay que actualizar *strings* gigantes manualmente en el backend de Python.
  * **Conclusión:** Funcional, pero inaceptable a nivel de arquitectura.

* **Iteración 2: Inyección Dinámica de JSON Schema (Pydantic)**
  * **Técnica:** Se usa `WorkspaceOutput.model_json_schema()` para inyectar dinámicamente el esquema OpenAPI estándar.
  * **Problema:** *Context Bloat* (Engorde del contexto). El estándar JSON Schema generó más de **1.100 tokens** de entrada debido a su verbosidad (`$defs`, metadatos).
  * **Impacto en Rendimiento:** La GPU tardó **5.4 segundos** de media en la fase de *Prefill*, penalizando severamente la UX.

* **Iteración 3: Inyección Dinámica de Esquema Minificado**
  * **Técnica:** Desarrollo de un algoritmo recursivo que comprime el esquema Pydantic, eliminando metadatos.
  * **Impacto en Rendimiento:** Los tokens de entrada se redujeron a **~450**, bajando el TTFT a **~2.2 segundos**, y recuperando la velocidad de generación a **~24.8 TPS**.

* **Iteración 4: Estrategia Definitiva (Interfaces TypeScript)**
  * **Técnica:** Implementación de un parser dinámico que traduce las clases Pydantic a interfaces nativas de TypeScript (`.d.ts`), aprovechando los Tipos Literales (`Literal` -> `widget_type: "kanban"`) para controlar el polimorfismo.
  * **Conclusión:** A nivel de los LLMs, TypeScript es un lenguaje más denso y natural para describir estructuras. Reduce el ruido de sintaxis, evita alucinaciones de tipado y mantiene una alineación arquitectónica perfecta entre el backend y el frontend de TaskFlow.

---

## 2. La Problemática de Escalabilidad y Enrutamiento de Intenciones

TaskFlow es un entorno multipropósito. Inyectar las interfaces TypeScript de los widgets disponibles en cada petición generaría nuevamente *Context Bloat*, disparando la latencia y provocando alucinaciones cruzadas.

Para solucionarlo, se diseñó el **Paso 1: Clasificador de Intenciones (Router)**, un llamado optimizado para evaluar qué widgets necesita el usuario antes de instanciarlos.

### Evaluación Empírica del Router
Se evaluó la capacidad deductiva del modelo frente a 5 escenarios.

| ID del Test | Complejidad | Decisión de la IA | Resultado | Latencia |
| :--- | :--- | :--- | :--- | :--- |
| **01-DIRECTO** | Explícita | `["kanban", "calendar"]` | Éxito (1:1) | 1.73s |
| **02-IMPLICITO** | Deductiva | `["kanban", "notes"]` | Éxito Lógico | 1.87s |
| **03-ANALITICA** | Compleja | `["grade_calculator", "pomodoro"]` | Éxito Lógico | 2.01s |
| **04-SOBRECARGA** | Múltiple | `["pomodoro", "notes", "calendar"]` | Éxito Total | 2.08s |
| **05-FUERA_DOMINIO**| Imposible | `[]` | Graceful Degradation | 1.66s |

**Análisis:** Se logró un **100% de acierto semántico y estructural**. El modelo demostró capacidad para mapear problemas abstractos a herramientas concretas y rechazar tareas fuera de dominio. *(Nota: En producción, con el catálogo en la KV Cache del servidor, la latencia descenderá previsiblemente a < 500ms).*

---

## 3. Comparativa de Modelos y el "Acantilado de la Inteligencia"

Se evaluó el comportamiento de la familia Qwen 2.5 en distintos tamaños de parámetros (14B, 7B y 3B) para observar la relación entre velocidad, consumo de RAM y capacidad de razonamiento.

* **El Milagro del 3B (Velocidad):** El modelo `Qwen2.5-3B` demostró un rendimiento técnico sin precedentes, superando los **100 TPS** con un consumo de apenas **2.3 GB de VRAM**. Para tareas de clasificación o generación simple, es excepcionalmente eficiente.
* **El Fallo del 3B (Format Confusion):** Al enfrentarse al nivel de estrés "Pesadilla" (TC-04), donde se requería instanciar múltiples widgets simultáneos, el modelo de 3B colapsó. Sufrió *Format Confusion*: al ver interfaces de TypeScript en su contexto, ignoró la orden de devolver JSON y comenzó a escribir código TypeScript funcional. Esto demuestra el **"Acantilado de la Inteligencia"**: los modelos pequeños pierden la capacidad de *Instruction Following* en contextos restrictivos complejos.
* **La Robustez del 7B/14B:** Los modelos de mayor tamaño poseían la capacidad cognitiva para separar el "medio" (plantilla TS) del "objetivo" (JSON), superando con éxito la prueba polimórfica.

---

## 4. Arquitectura Final: Orquestación Adaptativa (Advanced Model Cascading)

Los datos empíricos concluyen que no existe un modelo único que satisfaga simultáneamente los requisitos de latencia ultrabaja y razonamiento estructural complejo. Asignar estáticamente un modelo pesado para todas las tareas de generación supone un desperdicio de recursos computacionales (VRAM y batería).

Por ello, TaskFlow implementará un patrón de **Enrutamiento Dinámico Basado en Carga Cognitiva**, estructurado de la siguiente manera:

1. **Paso 1 - Enrutamiento y Triaje (Fast-Path):** Se utiliza siempre un modelo ultraligero (**3B parámetros**). Su tarea es leer la petición y devolver un array simple con las herramientas necesarias. Su baja latencia (< 500ms) garantiza una experiencia responsiva inmediata.
2. **Paso 2 - Evaluación Heurística (Middleware):** El Backend (FastAPI) evalúa determinísticamente la "Carga Cognitiva" de la petición basándose en la salida del Router y las variables del entorno:
   * *Nivel de Polimorfismo:* ¿Se solicita instanciar un solo widget o múltiples simultáneos?
   * *Volumen de Contexto:* ¿La petición incluye documentos externos recuperados vía RAG (> 1.000 tokens)?
3. **Paso 3 - Generación de UI Asimétrica:** Según la evaluación heurística, el sistema inyecta las interfaces TypeScript necesarias y selecciona dinámicamente el motor de generación óptimo:
   * **Ruta Ligera (Modelo 3B):** Si la tarea es simple (ej. un solo widget, contexto corto), se reutiliza el modelo de 3B. Se evitan los tiempos de carga en VRAM de modelos pesados, maximizando la velocidad de generación (~100 TPS) y minimizando el consumo energético.
   * **Ruta Pesada (Modelo 7B / 14B):** Si la tarea implica alta complejidad (multi-widget o RAG intensivo), el sistema escala la petición a un modelo superior. Este modelo asume un coste mayor de VRAM y latencia inicial (TTFT), pero previene empíricamente colapsos estructurales (*Format Confusion*) y garantiza un JSON perfectamente validado por Pydantic.

Esta arquitectura adaptativa exprime al máximo la Memoria Unificada (UMA) de Apple Silicon, asignando los ciclos de GPU estrictamente en proporción a la complejidad del problema, garantizando robustez a nivel de ingeniería y la latencia teórica más baja posible para cada interacción.

---

## 5. Conclusiones Test de Estrés: Viabilidad del Modelo 7B

La evaluación comparativa entre los modelos de 14B y 7B parámetros para la fase de Generación Estructural (Fase 2) revela que el modelo `Qwen2.5-7B-Instruct` representa el punto de equilibrio óptimo (*Sweet Spot*) para la arquitectura de TaskFlow. 

Los resultados empíricos demuestran que el modelo de 7B posee la capacidad cognitiva suficiente para comprender y aplicar las interfaces de TypeScript sin sufrir degradación estructural (*Format Confusion*), superando con éxito la validación estricta de Pydantic incluso bajo estrés de contexto masivo (10.000+ tokens). Al mismo tiempo, ofrece una mejora drástica en rendimiento bruto: el tiempo de latencia inicial (TTFT) se reduce a la mitad frente al modelo de 14B, y la velocidad de generación (TPS) se duplica, alcanzando los ~55 tokens/segundo en contextos ligeros.

Adicionalmente, el perfilado de memoria unificada en el chip M4 Pro confirma la eficiencia del modelo de 7B para un entorno de escritorio local. El consumo máximo de VRAM se estabilizó en **6.77 GB** durante la prueba de mayor estrés (frente a los casi 12 GB del modelo de 14B). Esta huella de memoria reducida es vital para el paradigma de *Advanced Model Cascading*, ya que permite mantener ambos modelos (el Router de 3B y el Generador de 7B) cargados simultáneamente en la memoria unificada de 24GB, dejando recursos computacionales holgados para el sistema operativo, el contenedor de Electron y la base de datos subyacente.

No obstante, la latencia de 28 segundos (TTFT) observada al inyectar el volcado completo de la base de datos (10.688 tokens) ratifica la conclusión previa: el procesamiento de contexto bruto es insostenible para una experiencia de usuario responsiva (*Bias for Action*). La implementación de un pipeline **RAG (Retrieval-Augmented Generation)** es una necesidad arquitectónica ineludible, con el fin de filtrar semánticamente el historial del workspace y acotar el prompt a un máximo de 1.500 tokens antes de invocar al motor de generación. Esto es necesario también para la primera decisión del Router ya que, con un contexto tan amplio, el modelo pierde precisión y devuelve todos los widgets disponibles.