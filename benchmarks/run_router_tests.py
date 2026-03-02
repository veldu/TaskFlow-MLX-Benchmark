import time
import json
import re
import pandas as pd
import mlx.core as mx
from mlx_lm import load, generate
import sys

MODEL_PATH = "mlx-community/Qwen2.5-14B-Instruct-4bit"

# Permitir sobreescritura externa (benchmark_suite.py puede inyectar CSV_OUTPUT)
CSV_OUTPUT = getattr(sys.modules[__name__], 'CSV_OUTPUT', "results/05_router_benchmark.csv")

# El catálogo de herramientas inyectado en el System Prompt
SYSTEM_PROMPT = """Eres el 'Router' del sistema operativo de productividad TaskFlow.
Tu única función es analizar el problema del usuario y decidir qué widgets de interfaz necesita para resolverlo.

# WIDGETS DISPONIBLES:
- "kanban": Organizar tareas, estados (To Do, Doing, Done), bugs o épicas.
- "calendar": Planificar fechas, ver plazos límite (deadlines) o eventos.
- "grade_calculator": Calcular notas medias, ponderaciones de asignaturas y simulaciones académicas.
- "pomodoro": Temporizador para estudiar o trabajar con concentración (focus).
- "notes": Tomar apuntes, redactar documentación o texto libre.

**REGLA CRÍTICA:** RESPONDE ÚNICA Y EXCLUSIVAMENTE CON UN ARRAY JSON DE STRINGS.
Ejemplo: ["kanban", "notes"]
Si el usuario pide algo imposible de hacer con estos widgets (ej. comprar cosas, buscar en internet), devuelve: []
"""

def extract_array(text: str) -> str:
    """Intenta limpiar la salida para asegurar que es parseable"""
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        return f"[{match.group(1)}]"
    return text.strip()

def main():
    print("==================================================")
    print("TASKFLOW: BENCHMARK DEL ROUTER (INTENT CLASSIFICATION)")
    print("==================================================")

    print(f"\nCargando modelo: {MODEL_PATH} ...")
    model, tokenizer = load(MODEL_PATH)
    
    print("Calentando GPU (Compilando grafos)...")
    generate(model, tokenizer, prompt="Hola", max_tokens=1, verbose=False)
    print("GPU lista. Empezando tests de enrutamiento.\n")

    with open("datasets/router_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for tc in test_cases:
        print(f"▶️ Ejecutando: {tc['id']} - {tc['name']}")
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nUsuario: \"{tc['prompt']}\"\n\nRespuesta JSON:"
        messages = [{"role": "user", "content": full_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        mx.reset_peak_memory()
        start_time = time.time()
        
        # Generación directa (sin streaming) porque es una respuesta ultracorta
        response_text = generate(model, tokenizer, prompt, max_tokens=30, verbose=False)
        
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        clean_response = extract_array(response_text)
        
        # Validación estricta
        is_valid_json = False
        try:
            parsed_array = json.loads(clean_response)
            if isinstance(parsed_array, list):
                is_valid_json = True
        except:
            pass
            
        peak_vram = round(mx.get_peak_memory() / (1024 * 1024), 2)
        
        results.append({
            "test_id": tc['id'],
            "latency_ms": latency_ms,
            "vram_mb": peak_vram,
            "raw_output": clean_response,
            "is_valid_json": is_valid_json,
            "expected_behavior": tc['expected_behavior']
        })
        
        status = "✅ JSON Válido" if is_valid_json else "❌ JSON Roto"
        print(f"   ⏱️ Latencia: {latency_ms} ms | 🧠 VRAM: {peak_vram} MB")
        print(f"   🤖 Decisión: {clean_response} ({status})\n")

    # Guardar resultados
    df = pd.DataFrame(results)
    # Resolver nombre de archivo: usar CSV_OUTPUT si se inyectó,
    # de lo contrario generarlo a partir del modelo cargado.
    out_file = CSV_OUTPUT if CSV_OUTPUT else f"results/{MODEL_PATH.split('/')[-1]}_router_benchmark.csv"
    df.to_csv(out_file, index=False)
    print(f"📊 Resultados guardados en {out_file}")

if __name__ == "__main__":
    main()
