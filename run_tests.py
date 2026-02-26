import json
import pandas as pd
from mlx_lm import load, generate
from scripts.benchmark_core import run_inference

MODEL_PATH = "mlx-community/Qwen2.5-14B-Instruct-4bit"

def main():
    print("==================================================")
    print("🚀 TASKFLOW: BENCHMARK DE INFERENCIA MLX")
    print("==================================================")
    print("Selecciona la estrategia de prueba:")
    print("  1. Baseline (Prompt original / Plantilla Manual)")
    print("  2. Schema Injection (JSON Schema completo)")
    print("  3. Minified Schema (Plantilla JSON Dinámica)")
    print("  4. TypeScript Interfaces (Token Diet + Dinámico)") # <--- NUEVA OPCIÓN
    print("==================================================")
    
    opcion = input("Elige una opción (1, 2, 3 o 4): ").strip()
    
    if opcion == "1":
        strategy = "baseline"
        output_csv = "results/01_baseline_report.csv"
    elif opcion == "2":
        strategy = "schema_injection"
        output_csv = "results/02_schema_injection_report.csv"
    elif opcion == "3":
        strategy = "minified_schema"
        output_csv = "results/03_minified_schema_report.csv"
    elif opcion == "4":
        strategy = "typescript_schema"
        output_csv = "results/04_typescript_schema_report.csv"
    else:
        print("❌ Opción no válida. Saliendo...")
        return

    print(f"\nCargando modelo: {MODEL_PATH} ...")
    model, tokenizer = load(MODEL_PATH)
    print("\nModelo cargado en Memoria.\n")

    print("Calentando la GPU (Compilando grafos de Metal)...")
    generate(model, tokenizer, prompt="Hola", max_tokens=1, verbose=False)
    print("GPU lista. Empezando tests reales.\n")

    with open("datasets/test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for tc in test_cases:
        print(f"▶️ Ejecutando: {tc['id']} - {tc['name']} (Nivel: {tc['level']})")
        
        metrics = run_inference(model, tokenizer, tc, strategy)
        
        metrics["model"] = MODEL_PATH
        metrics["strategy"] = strategy
        results.append(metrics)
        
        status = "✅ ÉXITO" if metrics["is_valid"] else f"❌ FALLO ({metrics['error_msg']})"
        print(f"   Resultado: {status} | In_Tokens: {metrics['input_tokens']} | TTFT: {metrics['ttft_ms']}ms | TPS: {metrics['tps']}\n")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"📊 Benchmark finalizado. Resultados guardados en: {output_csv}")

if __name__ == "__main__":
    main()