import os
import mlx.core as mx
from mlx_lm import load, generate

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def select_model() -> str:
    clear_console()
    print("==================================================")
    print("SELECCIÓN DE MODELO DE INFERENCIA")
    print("==================================================")
    models = [
        "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/Qwen3-14B-Instruct-4bit",
        "mlx-community/Qwen3-7B-Instruct-4bit",
        "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "Otro (Introducir ruta manual)"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("==================================================")
    
    opcion = input("Elige el modelo a evaluar (1-8): ").strip()
    
    if opcion == "8":
        return input("Introduce la ruta de HuggingFace (ej. mlx-community/modelo): ").strip()
    
    try:
        return models[int(opcion) - 1]
    except (ValueError, IndexError):
        print("Opción inválida. Usando Qwen2.5-14B por defecto.")
        return models[0]

def select_test_suite() -> str:
    print("\n==================================================")
    print("SELECCIÓN DE BATERÍA DE PRUEBAS")
    print("==================================================")
    print("  1. Generación de UI (Prompt Injection)")
    print("  2. Enrutamiento Lógico (Router / Intent Classification)")
    print("==================================================")
    
    return input("Elige la batería a ejecutar (1-2): ").strip()

def main():
    # 1. Selección dinámica del modelo
    model_path = select_model()
    
    # Normalizamos el nombre del modelo para usarlo en los archivos CSV
    model_safe_name = model_path.split("/")[-1] 
    
    # 2. Selección de la prueba
    test_suite = select_test_suite()
    
    print(f"\nCargando {model_safe_name} en Memoria Unificada...")
    
    # Guardamos el modelo seleccionado como variable de entorno para que los scripts hijos lo lean
    os.environ["TASKFLOW_MODEL_PATH"] = model_path
    os.environ["TASKFLOW_MODEL_NAME"] = model_safe_name

    clear_console()
    
    if test_suite == "1":
        print(f"Lanzando Test de Generación UI con {model_safe_name}...\n")
        import benchmarks.run_injection_tests as run_injection_tests
        run_injection_tests.MODEL_PATH = model_path
        # Actualizamos también el nombre seguro para que los ficheros generados lo lleven
        run_injection_tests.MODEL_SAFE_NAME = model_path.split("/")[-1]
        run_injection_tests.CSV_OUTPUT = os.environ.get("TASKFLOW_CSV_PATH", run_injection_tests.CSV_OUTPUT if hasattr(run_injection_tests, 'CSV_OUTPUT') else None)
        run_injection_tests.main()
        
    elif test_suite == "2":
        print(f"Lanzando Test del Router con {model_safe_name}...\n")
        import benchmarks.run_router_tests as run_router_tests
        run_router_tests.MODEL_PATH = model_path
        # Modificar dinámicamente el nombre del CSV en el script hijo
        csv_path = f"results/{model_safe_name}_router_benchmark.csv"
        run_router_tests.CSV_OUTPUT = csv_path
        run_router_tests.main()
        
    else:
        print("❌ Opción no válida. Saliendo...")

if __name__ == "__main__":
    main()