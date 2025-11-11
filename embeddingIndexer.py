import google.generativeai as genai
import faiss
import numpy as np
import json
import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# --- CONFIGURACIÓN ---

# Nombres de los archivos de almacenamiento
FAISS_INDEX_FILE = "faiss_index.index"
METADATA_MAP_FILE = "index_metadata_map.json"

# Modelo de embedding de Google AI
EMBEDDING_MODEL = "models/text-embedding-004"
# Dimensión de los embeddings (text-embedding-004 usa 768)
EMBEDDING_DIM = 768

# Modelo generativo de Google AI (para respuestas)
GENERATIVE_MODEL = "models/gemini-2.5-pro"

# Límite de la API: 100 documentos por llamada de embedding
BATCH_SIZE = 100

# --- FUNCIONES AUXILIARES ---

def load_chunks_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Carga los chunks desde un archivo .jsonl"""
    chunks = []
    print(f"Cargando chunks desde {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Leyendo .jsonl"):
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Advertencia: Omitiendo línea mal formada: {line}")
    return chunks

def get_embeddings_batched(
    chunks: List[Dict[str, Any]], 
    model: str
) -> np.ndarray:
    """
    Genera embeddings en lotes (batches) para una lista de chunks.
    Maneja reintentos simples para errores de API.
    """
    all_embeddings = []
    
    # Extraemos solo el contenido de texto para la API
    texts_to_embed = [chunk['content'] for chunk in chunks]
    
    print(f"Generando embeddings para {len(texts_to_embed)} chunks...")
    
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Generando Embeddings"):
        batch_texts = texts_to_embed[i:i + BATCH_SIZE]
        
        # Bucle de reintento simple
        retries = 3
        while retries > 0:
            try:
                # Genera el embedding para los documentos (RAG)
                result = genai.embed_content(
                    model=model,
                    content=batch_texts,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                all_embeddings.extend(result['embedding'])
                break # Éxito, salimos del bucle de reintento
            except Exception as e:
                print(f"\nError de API (reintento {4-retries}/3): {e}")
                retries -= 1
                time.sleep(5) # Espera 5 segundos antes de reintentar
        
        if retries == 0:
            raise Exception("Fallaron todos los reintentos de la API de embedding.")

    # FAISS requiere arrays de NumPy en formato float32
    return np.array(all_embeddings).astype('float32')

# --- MODOS DE OPERACIÓN (HANDLERS) ---

def handle_create(args):
    """
    Modo 'create': Borra cualquier índice existente y construye uno
    nuevo desde cero a partir de un *DIRECTORIO* de archivos .jsonl.
    """
    print(f"--- Modo CREATE: Reconstruyendo índice desde el directorio {args.input} ---")
    
    # 1. Borrar archivos antiguos si existen
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
        print(f"Archivo de índice antiguo borrado: {FAISS_INDEX_FILE}")
    if os.path.exists(METADATA_MAP_FILE):
        os.remove(METADATA_MAP_FILE)
        print(f"Archivo de metadatos antiguo borrado: {METADATA_MAP_FILE}")

    # --- Cargar desde directorio ---
    
    # 2. Cargar chunks desde todos los .jsonl en el directorio
    print(f"Buscando archivos .jsonl en: {args.input}")
    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: La ruta de entrada no es un directorio: {args.input}")
        return

    jsonl_files = list(input_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No se encontraron archivos .jsonl en {args.input}. Abortando.")
        return

    print(f"Se encontraron {len(jsonl_files)} archivos .jsonl. Cargando...")
    
    all_chunks = []
    # Usamos tqdm para la lista de archivos
    for file_path in tqdm(jsonl_files, desc="Procesando archivos"):
        chunks_from_file = load_chunks_from_jsonl(str(file_path))
        all_chunks.extend(chunks_from_file)
    
    chunks = all_chunks
    print(f"\nCarga completada. Total de chunks cargados: {len(chunks)}")

    if not chunks:
        print("No se cargaron chunks. Abortando.")
        return

    # 3. Generar embeddings
    embeddings = get_embeddings_batched(chunks, EMBEDDING_MODEL)
    
    if embeddings.shape[0] != len(chunks):
        print(f"Error: Discrepancia en el número de embeddings ({embeddings.shape[0]}) y chunks ({len(chunks)})")
        return
    
    # 4. Crear índice FAISS y añadir vectores
    print("Creando índice FAISS (IndexFlatL2)...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"Se añadieron {index.ntotal} vectores al índice.")

    # 5. Crear mapa de metadatos (ID de FAISS -> metadata del chunk)
    # Usamos str(i) porque las claves JSON deben ser strings
    metadata_map = {str(i): chunk for i, chunk in enumerate(chunks)}

    # 6. Guardar todo en disco
    print(f"Guardando índice en {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    print(f"Guardando metadatos en {METADATA_MAP_FILE}...")
    with open(METADATA_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_map, f, ensure_ascii=False, indent=2)
        
    print("\n¡Índice creado exitosamente!")

def handle_add(args):
    """
    Modo 'add': Carga un índice existente y añade nuevos
    documentos desde un archivo .jsonl.
    """
    print(f"--- Modo ADD: Añadiendo chunks desde {args.input} ---")

    # 1. Verificar que los archivos de índice existen
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_MAP_FILE):
        print(f"Error: No se encontró {FAISS_INDEX_FILE} o {METADATA_MAP_FILE}.")
        print("Ejecuta el modo 'create' primero.")
        return

    # 2. Cargar índice y metadatos existentes
    print("Cargando índice existente...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    
    print("Cargando metadatos existentes...")
    with open(METADATA_MAP_FILE, 'r', encoding='utf-8') as f:
        metadata_map = json.load(f)
    
    # El nuevo ID comenzará después del último ID existente
    start_index = index.ntotal
    
    # 3. Cargar nuevos chunks
    new_chunks = load_chunks_from_jsonl(args.input)
    if not new_chunks:
        print("No se cargaron nuevos chunks. Abortando.")
        return

    # 4. Generar embeddings para los *nuevos* chunks
    new_embeddings = get_embeddings_batched(new_chunks, EMBEDDING_MODEL)

    # 5. Añadir al índice
    print(f"Añadiendo {len(new_embeddings)} nuevos vectores al índice...")
    index.add(new_embeddings)

    # 6. Actualizar mapa de metadatos
    for i, chunk in enumerate(new_chunks):
        new_id = str(start_index + i)
        metadata_map[new_id] = chunk
    
    # 7. Guardar todo en disco
    print(f"Guardando índice actualizado en {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    print(f"Guardando metadatos actualizados en {METADATA_MAP_FILE}...")
    with open(METADATA_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_map, f, ensure_ascii=False, indent=2)
        
    print(f"\n¡Índice actualizado! Total de vectores ahora: {index.ntotal}")

def handle_search(args):
    """
    Modo 'search': Carga el índice y busca los k chunks
    más relevantes para una consulta.
    """
    print(f"--- Modo SEARCH: Buscando '{args.query}' ---")
    
    # 1. Verificar que los archivos de índice existen
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_MAP_FILE):
        print(f"Error: No se encontró {FAISS_INDEX_FILE} o {METADATA_MAP_FILE}.")
        print("Ejecuta el modo 'create' primero.")
        return

    # 2. Cargar índice y metadatos
    print("Cargando índice...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    
    print("Cargando metadatos...")
    with open(METADATA_MAP_FILE, 'r', encoding='utf-8') as f:
        metadata_map = json.load(f)

    # 3. Generar embedding para la CONSULTA (query)
    print("Generando embedding para la consulta...")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=args.query,
            task_type="RETRIEVAL_QUERY" # ¡Importante! 'RETRIEVAL_QUERY' para la búsqueda
        )
        query_vector = np.array([result['embedding']]).astype('float32')
    except Exception as e:
        print(f"Error al generar embedding para la consulta: {e}")
        return

    # 4. Buscar en FAISS
    k = args.k
    print(f"Buscando los {k} vecinos más cercanos...")
    distances, indices = index.search(query_vector, k)

    # 5. Mostrar resultados
    print("\n--- Resultados de la Búsqueda ---")
    if not indices.size:
        print("No se encontraron resultados.")
        return

    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        # Usamos str(idx) para buscar en el mapa de metadatos
        chunk_data = metadata_map.get(str(idx))
        
        if chunk_data:
            metadata = chunk_data['metadata']
            print(f"\nResultado {i+1} (ID: {idx}, Distancia: {dist:.4f})")
            print(f"  Fuente: {metadata.get('source_file')}")
            print(f"  Sección: {metadata.get('logical_section')}")
            print("  Contenido:")
            print(f"  > {chunk_data['content'][:400]}...") # Muestra un fragmento
        else:
            print(f"\nResultado {i+1} (ID: {idx}) - No se encontraron metadatos.")

# --- FUNCIÓN 'ASK' (RAG) ---

def handle_ask(args):
    """
    Modo 'ask': Realiza una búsqueda RAG.
    1. Recupera chunks (Retrieval)
    2. Genera una respuesta basada en los chunks (Generation)
    """
    print(f"--- Modo ASK: Respondiendo a '{args.query}' ---")
    
    # 1. Verificar que los archivos de índice existen
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_MAP_FILE):
        print(f"Error: No se encontró {FAISS_INDEX_FILE} o {METADATA_MAP_FILE}.")
        print("Ejecuta el modo 'create' primero.")
        return

    # --- PASO 1: CAPA DE BÚSQUEDA (RETRIEVAL) ---
    
    print("Cargando índice y metadatos...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_MAP_FILE, 'r', encoding='utf-8') as f:
        metadata_map = json.load(f)

    print("Generando embedding para la consulta...")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=args.query,
            task_type="RETRIEVAL_QUERY"
        )
        query_vector = np.array([result['embedding']]).astype('float32')
    except Exception as e:
        print(f"Error al generar embedding para la consulta: {e}")
        return

    k = args.k
    print(f"Buscando los {k} chunks más relevantes...")
    distances, indices = index.search(query_vector, k)

    if not indices.size:
        print("No se encontraron documentos relevantes para responder la pregunta.")
        return

    # Recopilar los chunks de contexto
    context_chunks = []
    for idx in indices[0]:
        chunk_data = metadata_map.get(str(idx))
        if chunk_data:
            context_chunks.append(chunk_data)

    if not context_chunks:
        print("Se encontraron índices pero no metadatos. El mapa está corrupto.")
        return

    # --- PASO 2: CAPA DE GENERACIÓN (GENERATION) ---

    print("Construyendo prompt de contexto y generando respuesta...")

    # 2.1. Construir el prompt de contexto
    
    # Instrucción de sistema (System Prompt)
    system_prompt = (
        "Eres un asistente amable y servicial para familias. "
        "Tu tarea es responder la pregunta del usuario basándote *única y exclusivamente* "
        "en los siguientes extractos de documentos (contexto) que se te proporcionan.\n"
        "Redacta una respuesta clara, concisa y en un lenguaje fácil de entender.\n"
        "Cuando sea posible, cita la fuente del documento (ej. 'Según la Guía X...') "
        "usando la información de 'Fuente:' y 'Sección:' de los extractos."
    )
    
    # Formatear los chunks de contexto
    context_str = "--- INICIO DEL CONTEXTO ---\n"
    for i, chunk in enumerate(context_chunks):
        metadata = chunk['metadata']
        context_str += (
            f"\nExtracto {i+1}:\n"
            f"Fuente: {metadata.get('source_file')}\n"
            f"Sección: {metadata.get('logical_section')}\n"
            f"Contenido: {chunk['content']}\n"
            f"--- (fin del extracto {i+1}) ---\n"
        )
    context_str += "--- FIN DEL CONTEXTO ---\n"
    
    # Prompt final
    final_prompt = (
        f"{system_prompt}\n\n"
        f"{context_str}\n\n"
        f"Pregunta del Usuario: {args.query}\n\n"
        f"Respuesta (basada *sólo* en el contexto anterior):"
    )

    # 2.2. Llamar al modelo generativo
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        response = model.generate_content(final_prompt)
        
        # 2.3. Mostrar la respuesta
        print("\n--- Respuesta Generada ---")
        print(response.text)
        print("--------------------------")

    except Exception as e:
        print(f"\nError al generar la respuesta: {e}")
        print("\n--- Contexto que se iba a utilizar ---")
        print(final_prompt)

# --- PUNTO DE ENTRADA PRINCIPAL ---

def main():
    # 1. Configurar la API de Google
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
        print("Por favor, configúrala antes de ejecutar el script.")
        return
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error al configurar la API de Google: {e}")
        return

    # 2. Configurar ArgumentParser (CLI)
    parser = argparse.ArgumentParser(
        description="Indexador RAG con Google AI (Gemini) y FAISS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Comando 'create' ---
    create_help = (
        "Crea (o reconstruye) el índice FAISS desde cero.\n"
        "Borra cualquier índice y mapa de metadatos existentes.\n"
        "Espera un *directorio* que contenga los archivos .jsonl.\n"
        "**ESTRATEGIA DE ACTUALIZACIÓN:** Si una normativa cambia,\n"
        "vuelve a generar tus .jsonl y usa 'create' para recalcular todo."
    )
    create_parser = subparsers.add_parser(
        'create', 
        help="Reconstruye el índice completo desde un directorio.",
        description=create_help
    )
    create_parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Directorio de entrada con archivos .jsonl (ej. ./chunks_individuales)"
    )
    create_parser.set_defaults(func=handle_create)

    # --- Comando 'add' ---
    add_parser = subparsers.add_parser(
        'add', 
        help="Añade nuevos documentos a un índice existente."
    )
    add_parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Archivo .jsonl con los *nuevos* chunks a añadir."
    )
    add_parser.set_defaults(func=handle_add)

    # --- Comando 'search' ---
    search_parser = subparsers.add_parser(
        'search', 
        help="Busca en el índice con una consulta de texto."
    )
    search_parser.add_argument(
        '--query', 
        type=str, 
        required=True, 
        help="El texto que quieres buscar."
    )
    search_parser.add_argument(
        '-k', 
        type=int, 
        default=5, 
        help="Número de resultados a devolver (default: 5)"
    )
    search_parser.set_defaults(func=handle_search)
    
    # --- Comando 'ask' ---
    ask_parser = subparsers.add_parser(
        'ask',
        help="Realiza una respuesta dada la información de la pregunta"
    )
    ask_parser.add_argument(
        '--query',
        required=True,
        help="Pregunta a realizar"
    )
    ask_parser.add_argument(
        '-k', 
        type=int, 
        default=5, 
        help="Número de resultados a devolver (default: 5)"
    )
    ask_parser.set_defaults(func=handle_ask)
    
    # 3. Parsear argumentos y ejecutar la función
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()