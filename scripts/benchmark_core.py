import time
import json
import re
import mlx.core as mx
from mlx_lm import stream_generate
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Optional, Any

# ==========================================
# 1. SCHEMAS DE PYDANTIC (Generative UI)
# ==========================================

class ChecklistItem(BaseModel):
    task: str
    is_completed: bool = False

class Task(BaseModel):
    id: str = Field(description="Identificador único generado por IA")
    title: str
    description: Optional[str] = None
    priority: Literal["low", "medium", "high"]
    checklist: Optional[List[ChecklistItem]] = []

class KanbanColumn(BaseModel):
    id: str
    title: str = Field(description="Ej: 'To Do', 'In Progress', 'Done', o personalizadas")
    tasks: List[Task]

class KanbanWidget(BaseModel):
    widget_type: Literal["kanban"] = "kanban"
    title: str
    columns: List[KanbanColumn]

class WorkspaceOutput(BaseModel):
    intent_summary: str = Field(description="Breve resumen de lo que la IA ha entendido que el usuario necesita")
    widgets: List[KanbanWidget]

# ==========================================
# 2. FUNCIONES DE OPTIMIZACIÓN DE ESQUEMAS
# ==========================================

def _build_mock_dict(schema: dict, defs: dict) -> Any:
    """Convierte JSON Schema en plantilla JSON."""
    if '$ref' in schema:
        ref_name = schema['$ref'].split('/')[-1]
        return _build_mock_dict(defs[ref_name], defs)
    elif schema.get('type') == 'object':
        props = schema.get('properties', {})
        return {k: _build_mock_dict(v, defs) for k, v in props.items()}
    elif schema.get('type') == 'array':
        items = schema.get('items', {})
        return [_build_mock_dict(items, defs)]
    elif 'anyOf' in schema:
        for choice in schema['anyOf']:
            if choice.get('type') != 'null':
                return _build_mock_dict(choice, defs)
        return "any"
    else:
        t = schema.get('type', 'any')
        if 'enum' in schema:
            return "|".join(schema['enum'])
        return t

def get_minified_schema() -> str:
    raw_schema = WorkspaceOutput.model_json_schema()
    defs = raw_schema.get('$defs', {})
    compact_dict = _build_mock_dict(raw_schema, defs)
    return json.dumps(compact_dict, indent=2)

def get_typescript_schema() -> str:
    """Traduce dinámicamente Pydantic OpenAPI Schema a TypeScript Interfaces."""
    schema = WorkspaceOutput.model_json_schema()
    defs = schema.get('$defs', {})
    ts_interfaces = []

    def resolve_type(prop_info):
        if '$ref' in prop_info:
            return prop_info['$ref'].split('/')[-1]
        if 'anyOf' in prop_info:
            types = [resolve_type(t) for t in prop_info['anyOf'] if t.get('type') != 'null']
            return " | ".join(types) if types else "any"
        
        t = prop_info.get('type', 'any')
        if t == 'string':
            if 'enum' in prop_info:
                return " | ".join([f'"{e}"' for e in prop_info['enum']])
            return 'string'
        elif t in ['integer', 'number']:
            return 'number'
        elif t == 'boolean':
            return 'boolean'
        elif t == 'array':
            item_type = resolve_type(prop_info.get('items', {}))
            return f"{item_type}[]"
        elif t == 'object':
            return 'Record<string, any>'
        return 'any'

    # Generar interfaces hijas
    for model_name, model_info in defs.items():
        props = model_info.get('properties', {})
        required = model_info.get('required', [])
        lines = [f"interface {model_name} {{"]
        for prop_name, prop_data in props.items():
            req_mark = "" if prop_name in required else "?"
            ts_type = resolve_type(prop_data)
            lines.append(f"  {prop_name}{req_mark}: {ts_type};")
        lines.append("}")
        ts_interfaces.append("\n".join(lines))

    # Generar interfaz root
    props = schema.get('properties', {})
    required = schema.get('required', [])
    lines = [f"interface WorkspaceOutput {{"]
    for prop_name, prop_data in props.items():
        req_mark = "" if prop_name in required else "?"
        ts_type = resolve_type(prop_data)
        lines.append(f"  {prop_name}{req_mark}: {ts_type};")
    lines.append("}")
    ts_interfaces.append("\n".join(lines))

    return "\n\n".join(ts_interfaces)

# ==========================================
# 3. FUNCIONES CORE DEL BENCHMARK
# ==========================================

def extract_json(text: str) -> str:
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

def validate_response(json_str: str, test_id: str) -> tuple[bool, str]:
    try:
        data = json.loads(json_str)
        WorkspaceOutput(**data)
        return True, "Validación Exitosa (Árbol JSON correcto)"
    except json.JSONDecodeError as e:
        return False, f"JSON Roto (Sintaxis): {str(e)}"
    except ValidationError as e:
        error_path = " -> ".join([str(loc) for loc in e.errors()[0]['loc']])
        return False, f"Schema Inválido en [{error_path}]: {e.errors()[0]['msg']}"
    except Exception as e:
        return False, f"Error desconocido: {str(e)}"

def run_inference(model, tokenizer, test_case: dict, strategy: str) -> dict:
    mx.reset_peak_memory()
    
    # --- APLICACIÓN DE LA ESTRATEGIA ---
    if strategy == "schema_injection":
        schema_json = WorkspaceOutput.model_json_schema()
        schema_str = json.dumps(schema_json, indent=2)
        system_prompt_final = (
            f"{test_case['system_prompt']}\n\n"
            f"CRÍTICO: Tu salida DEBE ser un JSON válido que cumpla estrictamente con el siguiente JSON Schema:\n"
            f"```json\n{schema_str}\n```\n"
        )
    elif strategy == "minified_schema":
        schema_str = get_minified_schema()
        system_prompt_final = (
            f"{test_case['system_prompt']}\n\n"
            f"CRÍTICO: Tu salida DEBE ser un JSON válido que cumpla estrictamente con esta ESTRUCTURA:\n"
            f"```json\n{schema_str}\n```\n"
        )
    elif strategy == "typescript_schema":
        schema_str = get_typescript_schema()
        system_prompt_final = (
            f"{test_case['system_prompt']}\n\n"
            f"CRÍTICO: Tu salida DEBE ser un JSON puramente de datos que implemente estas interfaces TypeScript:\n"
            f"```typescript\n{schema_str}\n```\n"
        )
    else:
        system_prompt_final = test_case['system_prompt']

    full_prompt = f"{system_prompt_final}\n\nContexto: {test_case['context']}\n\nUsuario: {test_case['prompt']}"
    
    messages = [{"role": "user", "content": full_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    input_tokens_count = len(tokenizer.encode(prompt))
    
    start_time = time.time()
    first_token_time = None
    generated_text = ""
    output_tokens_count = 0
    
    for response in stream_generate(model, tokenizer, prompt, max_tokens=1024):
        if first_token_time is None:
            first_token_time = time.time()
        generated_text += response.text
        output_tokens_count += 1
        
    end_time = time.time()
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_time_s = end_time - start_time
    tps = output_tokens_count / total_time_s if total_time_s > 0 else 0
    
    peak_vram_mb = mx.get_peak_memory() / (1024 * 1024)

    clean_json = extract_json(generated_text)
    is_valid, error_msg = validate_response(clean_json, test_case['id'])

    return {
        "test_id": test_case['id'],
        "input_tokens": input_tokens_count,
        "ttft_ms": round(ttft_ms, 2),
        "tps": round(tps, 2),
        "total_time_s": round(total_time_s, 2),
        "tokens_generated": output_tokens_count,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "is_valid": is_valid,
        "error_msg": error_msg,
        "raw_output": generated_text
    }