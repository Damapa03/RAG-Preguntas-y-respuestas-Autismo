import fitz  # Importa PyMuPDF
import os
from pathlib import Path

# --- CONFIGURACIÓN ---
# 1. Indica la carpeta que contiene tus archivos PDF
CARPETA_PDFS = Path("H:/Mi unidad/Classroom/26PIA/Proyecto RAG/texto/pdf")

# 2. Indica la carpeta donde quieres guardar los archivos TXT
CARPETA_TXT = Path("H:/Mi unidad/Classroom/26PIA/Proyecto RAG/texto/txt")
# ---------------------

def extraer_texto_de_pdfs(ruta_entrada, ruta_salida):
    """
    Recorre la carpeta de entrada, extrae el texto de cada PDF
    y lo guarda en la carpeta de salida.
    """
    
    # 1. Asegurarse de que la carpeta de salida exista
    # Si no existe, la crea.
    os.makedirs(ruta_salida, exist_ok=True)
    
    print(f"Buscando PDFs en: {ruta_entrada}")
    print(f"Guardando TXTs en: {ruta_salida}\n")
    
    # 2. Recorrer todos los archivos en la carpeta de entrada
    # .glob("*.pdf") solo selecciona archivos que terminan en .pdf
    archivos_pdf_encontrados = list(ruta_entrada.glob("*.pdf"))
    
    if not archivos_pdf_encontrados:
        print("Advertencia: No se encontró ningún archivo PDF en la carpeta de entrada.")
        return

    for ruta_pdf in archivos_pdf_encontrados:
        texto_completo = ""
        
        try:
            # 3. Abrir el documento PDF
            with fitz.open(ruta_pdf) as doc:
                # 4. Extraer el texto de cada página
                for pagina in doc:
                    texto_completo += pagina.get_text()
            
            # 5. Crear el nombre y la ruta para el archivo .txt
            # ruta_pdf.stem -> "documento_importante" (si el archivo es "documento_importante.pdf")
            nombre_txt = ruta_pdf.stem + ".txt"
            ruta_txt_salida = ruta_salida / nombre_txt
            
            # 6. Guardar el texto extraído en el archivo .txt
            # Se usa encoding="utf-8" para asegurar compatibilidad con tildes y caracteres especiales
            with open(ruta_txt_salida, "w", encoding="utf-8") as f_out:
                f_out.write(texto_completo)
                
            print(f"✅ Procesado: {ruta_pdf.name}  ->  {nombre_txt}")

        except Exception as e:
            # Capturar cualquier error que ocurra con un PDF (ej. si está corrupto o protegido)
            print(f"❌ Error al procesar {ruta_pdf.name}: {e}")

    print("\n¡Proceso completado!")

# --- Ejecutar la función ---
if __name__ == "__main__":
    extraer_texto_de_pdfs(CARPETA_PDFS, CARPETA_TXT)