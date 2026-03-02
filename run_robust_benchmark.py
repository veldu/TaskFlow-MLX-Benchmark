import os
import time
import json
import sys
import pandas as pd
import mlx.core as mx
from mlx_lm import load, stream_generate, generate

from scripts.benchmark_core import get_typescript_schema, WorkspaceOutput, extract_json, validate_response

# variable CSV de salida reutilizable (similar a otros tests)
CSV_OUTPUT = getattr(sys.modules[__name__], 'CSV_OUTPUT', None)


# ==========================================
# 0. MENÚS INTERACTIVOS
# ==========================================
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def select_router_model() -> str:
    clear_console()
    print("==================================================")
    print("SELECCIÓN DE MOTOR DE ENRUTAMIENTO (FASE 1)")
    print("==================================================")
    print("Selecciona un modelo ultraligero (< 4B) para clasificar intenciones:\n")
    
    models = [
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "Otro (Introducir ruta manual)"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("==================================================")
    
    opcion = input("Elige el modelo Router (1-5): ").strip()
    
    if opcion == str(len(models)):
        return input("Introduce la ruta de HuggingFace (ej. mlx-community/modelo): ").strip()
    
    try:
        return models[int(opcion) - 1]
    except (ValueError, IndexError):
        print("[SISTEMA] Opción inválida. Usando Qwen2.5-3B por defecto.")
        return models[0]

def select_generator_model() -> str:
    clear_console()
    print("==================================================")
    print("SELECCIÓN DE MOTOR DE GENERACIÓN (FASE 2)")
    print("==================================================")
    print("Selecciona el modelo pesado (>= 3.8B) para instanciar la UI:\n")
    
    models = [
        "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/gemma-2-9b-it-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "Otro (Introducir ruta manual)"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("==================================================")
    
    opcion = input("Elige el modelo Generador (1-7): ").strip()
    
    if opcion == str(len(models)):
        return input("Introduce la ruta de HuggingFace (ej. mlx-community/modelo): ").strip()
    
    try:
        return models[int(opcion) - 1]
    except (ValueError, IndexError):
        print("[SISTEMA] Opción inválida. Usando Qwen2.5-7B por defecto.")
        return models[1]

# ==========================================
# 1. GENERADOR DE CONTEXTO REALISTA
# ==========================================
def generate_db_json_context(multiplier: int) -> str:
    lorem_markdown = (
        "## Discusión sobre Arquitectura\n\n"
        "Durante la última revisión del proyecto, se decidió establecer Pydantic "
        "como la Única Fuente de Verdad (SSOT). Esto implica que cualquier cambio "
        "en las interfaces del frontend deberá originarse en los esquemas de Python. "
        "Además, la latencia de inferencia observada con el framework MLX requiere "
        "una optimización estricta del prompt.\n\n"
        "* Punto 1: Implementar generador de tipos.\n"
        "* Punto 2: Perfilar consumo de memoria en M4 Pro.\n"
    )

    base_state = {
        "workspace_id": "ws_alpha_01",
        "last_updated": "2026-03-02T12:00:00Z",
        "widgets": [
            {
                "widget_type": "kanban",
                "title": "Historial de Desarrollo TFG",
                "columns": [
                    {
                        "id": "col_backlog_01",
                        "title": "BACKLOG",
                        "tasks": [
                            {
                                "id": "task_101",
                                "title": "Investigación de Modelos Cuantizados",
                                "description": "Analizar la familia Qwen 2.5 y comparar rendimientos.",
                                "priority": "low",
                                "checklist": [
                                    {"task": "Descargar Qwen2.5-3B-Instruct", "is_completed": True},
                                    {"task": "Descargar Qwen2.5-14B-Instruct", "is_completed": True},
                                    {"task": "Ejecutar benchmark de perplexity", "is_completed": False}
                                ]
                            }
                        ]
                    },
                    {
                        "id": "col_done_01",
                        "title": "DONE",
                        "tasks": [
                            {
                                "id": "task_201",
                                "title": "Configuración del Monorepo",
                                "description": "Inicializar Vite, React, Tailwind, Electron y FastAPI.",
                                "priority": "medium",
                                "checklist": []
                            }
                        ]
                    }
                ]
            },
            {
                "widget_type": "notes",
                "title": "Acta de Reunión de Seguimiento",
                "content": lorem_markdown * 3 
            }
        ]
    }
    
    massive_state = {"workspace_id": "ws_alpha_01", "widgets": []}
    
    for i in range(multiplier):
        for widget in base_state["widgets"]:
            widget_clone = json.loads(json.dumps(widget))
            widget_clone["title"] = f"{widget_clone['title']} (Volumen {i})"
            massive_state["widgets"].append(widget_clone)
            
    return json.dumps(massive_state, indent=2)

# ==========================================
# 2. PROMPT DEL USUARIO
# ==========================================
COMPLEX_USER_PROMPT = """
A ver, tengo la entrega del TFG en 3 semanas y necesito organizarme. 
Quiero un tablero para gestionar lo que me queda: terminar el benchmark del router (alta prioridad), 
escribir la memoria (media) y corregir tests de Pytest que fallan (alta). 
Hazme otra columna de 'En revisión' por si el tutor pide cambios.
Aparte, necesito un espacio de notas en blanco para apuntar el feedback del tribunal.
Ignora las tareas viejas del workspace, céntrate solo en esto nuevo.
"""

# ==========================================
# 3. MOTOR DEL TEST DE ESTRÉS E2E
# ==========================================
def run_e2e_stress_test(models: dict, context_multiplier: int, test_name: str):
    print(f"\n[{test_name}] INICIANDO PRUEBA...")
    
    router_model, router_tok = models["router"]
    gen_model, gen_tok = models["generator"]
    
    db_context_json = generate_db_json_context(context_multiplier)
    
    # --- FASE 1 ---
    print(f"[{test_name}] [FASE 1] Ejecutando clasificador de intenciones...")
    
    router_system = (
        "Eres el Router de TaskFlow. Analiza la petición del usuario y el contexto de la base de datos. "
        "Devuelve ÚNICAMENTE un array JSON con los identificadores de los widgets necesarios: "
        "['kanban', 'notes', 'calendar', 'grade_calculator', 'pomodoro']."
    )
    router_prompt = f"{router_system}\n\n[CONTEXTO DB (JSON)]\n{db_context_json}\n\n[USUARIO]\n{COMPLEX_USER_PROMPT}"
    
    messages = [{"role": "user", "content": router_prompt}]
    formatted_router_prompt = router_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    start_time_router = time.time()
    router_response = generate(router_model, router_tok, formatted_router_prompt, max_tokens=30, verbose=False)
    router_time_ms = (time.time() - start_time_router) * 1000
    
    print(f"[{test_name}] [FASE 1] Latencia Router: {router_time_ms:.2f} ms")
    print(f"[{test_name}] [FASE 1] Decisión Router: {router_response.strip()}")
    
    # --- FASE 2 ---
    print(f"[{test_name}] [FASE 2] Escalando a motor de generación pesado...")
    
    ts_schema = get_typescript_schema()
    generator_system = (
        "Eres el motor lógico de TaskFlow. Genera la estructura de datos para la interfaz solicitada.\n"
        "REGLAS CRÍTICAS:\n"
        "1. La salida DEBE ser un único JSON válido.\n"
        "2. El JSON generado debe cumplir estrictamente con las siguientes interfaces TypeScript:\n"
        f"```typescript\n{ts_schema}\n```\n"
        "No incluyas explicaciones en texto natural, solo el bloque JSON."
    )
    
    generator_prompt = f"{generator_system}\n\n[CONTEXTO DB ACTUAL]\n{db_context_json}\n\n[PETICIÓN USUARIO]\n{COMPLEX_USER_PROMPT}"
    
    messages = [{"role": "user", "content": generator_prompt}]
    formatted_gen_prompt = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    input_tokens = len(gen_tok.encode(formatted_gen_prompt))
    print(f"[{test_name}] [FASE 2] Tamaño del Contexto Inyectado: {input_tokens} tokens")
    
    mx.reset_peak_memory()
    start_time_gen = time.time()
    first_token_time = None
    generated_text = ""
    output_tokens = 0
    
    for response in stream_generate(gen_model, gen_tok, formatted_gen_prompt, max_tokens=1500):
        if first_token_time is None:
            first_token_time = time.time()
            ttft_ms = (first_token_time - start_time_gen) * 1000
            print(f"[{test_name}] [FASE 2] Prefill completado. TTFT: {ttft_ms:.2f} ms")
        
        generated_text += response.text
        output_tokens += 1
        
    total_gen_time = time.time() - start_time_gen
    tps = output_tokens / (time.time() - first_token_time) if first_token_time else 0
    peak_vram = mx.get_peak_memory() / (1024 * 1024 * 1024)
    
    clean_json = extract_json(generated_text)
    is_valid, error_msg = validate_response(clean_json, test_name)
    
    print(f"\n[{test_name}] --- RESULTADOS GLOBALES ---")
    print(f"[{test_name}] TTFT (Latencia Inicial) : {ttft_ms:.2f} ms")
    print(f"[{test_name}] TPS (Velocidad Decoding): {tps:.2f} tokens/seg")
    print(f"[{test_name}] Consumo VRAM (Pico)     : {peak_vram:.2f} GB")
    print(f"[{test_name}] Validación Estructural  : {'EXITOSA' if is_valid else 'FALLIDA'}")
    
    if not is_valid:
        print(f"[{test_name}] Error de Validación     : {error_msg}")
        
    print(f"[{test_name}] Payload Parcial Generado:")
    print(f"{clean_json[:400]}...\n[FIN DEL TEST]\n")

    # devolver métricas para el CSV
    return {
        "test_name": test_name,
        "router_latency_ms": router_time_ms,
        "context_tokens": input_tokens,
        "ttft_ms": ttft_ms,
        "tps": tps,
        "peak_vram_gb": peak_vram,
        "validation_success": is_valid,
        "validation_error": error_msg if not is_valid else "",
        "router_model": models["router"][0].name if hasattr(models["router"][0], 'name') else '',
        "generator_model": models["generator"][0].name if hasattr(models["generator"][0], 'name') else ''
    }

# ==========================================
# 4. ORQUESTADOR PRINCIPAL
# ==========================================
def main():
    # 1. Solicitar modelos dinámicamente
    router_model_path = select_router_model()
    generator_model_path = select_generator_model()
    
    clear_console()
    print("[SISTEMA] Iniciando suite de Benchmark Avanzado (Cascading Models).")
    
    # 2. Cargar modelo Router
    print(f"[SISTEMA] Cargando modelo Router ({router_model_path})...")
    router_model, router_tokenizer = load(router_model_path)
    
    # 3. Cargar modelo Generador
    print(f"[SISTEMA] Cargando modelo Generador ({generator_model_path})...")
    gen_model, gen_tokenizer = load(generator_model_path)
    
    models = {
        "router": (router_model, router_tokenizer),
        "generator": (gen_model, gen_tokenizer)
    }
    
    print("[SISTEMA] Calentando GPU (Compilando grafos MLX)...")
    generate(router_model, router_tokenizer, prompt="test", max_tokens=1, verbose=False)
    generate(gen_model, gen_tokenizer, prompt="test", max_tokens=1, verbose=False)
    print("[SISTEMA] GPU lista.\n")
    print("-" * 60)
    
    results = []
    res1 = run_e2e_stress_test(models, context_multiplier=2, test_name="ESTRÉS NIVEL 1 (Workspace Ligero)")
    res1["router_model"] = router_model_path
    res1["generator_model"] = generator_model_path
    results.append(res1)
    res2 = run_e2e_stress_test(models, context_multiplier=10, test_name="ESTRÉS NIVEL 2 (Workspace Pesado)")
    res2["router_model"] = router_model_path
    res2["generator_model"] = generator_model_path
    results.append(res2)
    
    # guardar CSV final
    os.makedirs("results/robust", exist_ok=True)
    # decidir nombre de archivo
    safe_name = generator_model_path.split("/")[-1]
    default_csv = f"results/robust/{safe_name}_robust.csv"
    output_csv = CSV_OUTPUT or default_csv
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n Resultados guardados en: {output_csv}")

if __name__ == "__main__":
    main()