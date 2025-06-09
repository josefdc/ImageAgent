# En: tests/core/test_processing.py

import pytest
from PIL import Image, ImageChops, ImageDraw # <--- IMPORT AÑADIDO
import numpy as np
# Asegúrate de que el intérprete pueda encontrar tus módulos core, agent, etc.
# Esto podría requerir ajustar PYTHONPATH o la estructura del proyecto/pruebas.
from core.processing import apply_rotation, apply_brightness # Añade otras funciones a medida que las pruebes

# --- Fixture de Imagen de Prueba ---
@pytest.fixture(scope="module") # scope="module" para crear la imagen una sola vez por módulo
def reference_image():
    """Crea una imagen de referencia más útil para pruebas de rotación."""
    img = Image.new('RGB', (50, 100), color='lightgrey') # Rectangular para notar cambios de tamaño
    draw = ImageDraw.Draw(img) # <--- Ahora ImageDraw está definido
    # Un punto distintivo en cada cuadrante (relativo al tamaño 50x100)
    draw.point((10, 10), fill=(255, 0, 0))    # Top-Left (Rojo)
    draw.point((40, 10), fill=(0, 0, 255))   # Top-Right (Azul)
    draw.point((10, 90), fill=(0, 255, 0))  # Bottom-Left (Verde)
    draw.point((40, 90), fill=(255, 255, 0)) # Bottom-Right (Amarillo)
    return img

# --- Tests para apply_rotation (Parametrizados) ---
@pytest.mark.parametrize(
    "angle, expected_size, expected_top_left_color_at",
    [
        (0, (50, 100), ((10, 10), (255, 0, 0))),   # Sin rotación, punto TL rojo en (10,10)
        (90, (100, 50), ((10, 40), (255, 0, 0))),  # Rota 90, punto TL rojo en (10, 40) (coord Y, W-1-coord X)
        (180, (50, 100), ((40, 90), (255, 0, 0))), # Rota 180, punto TL rojo en (40, 90) (W-1-X, H-1-Y)
        (270, (100, 50), ((90, 10), (255, 0, 0))), # Rota 270, punto TL rojo en (90, 10) (H-1-Y, X)
        (360, (50, 100), ((10, 10), (255, 0, 0))), # Rotación completa
        (-90, (100, 50), ((90, 10), (255, 0, 0))), # Equivalente a 270
    ],
    ids=["0deg", "90deg", "180deg", "270deg", "360deg", "-90deg"] # IDs para mejor legibilidad
)
def test_apply_rotation_angles(reference_image, angle, expected_size, expected_top_left_color_at):
    """Verifica rotaciones con diferentes ángulos usando parametrización."""
    rotated_image = apply_rotation(reference_image, angle)
    coord, expected_rgb = expected_top_left_color_at

    assert rotated_image is not None, f"Rotación con ángulo {angle} no debería ser None"
    assert rotated_image.mode == reference_image.mode, f"Modo incorrecto tras rotar {angle} grados"
    # La implementación usa expand=True, así que el tamaño cambia para 90/270 en imágenes no cuadradas
    assert rotated_image.size == expected_size, f"Tamaño incorrecto tras rotar {angle} grados"

    # Verificar que el píxel que *era* el rojo original ahora está en la nueva coordenada
    # y tiene el color rojo. Usamos una pequeña tolerancia por el resampling BICUBIC.
    actual_pixel_rgb = rotated_image.getpixel(coord)
    assert np.allclose(actual_pixel_rgb, expected_rgb, atol=10), \
        f"Pixel en {coord} tras rotar {angle}° no era {expected_rgb}. Fue: {actual_pixel_rgb}"

    # Verificar si realmente hubo cambio (excepto para 0 y 360 grados)
    if angle % 360 != 0:
        # Redimensionar la original al tamaño de la rotada para comparar si cambió el tamaño
        resized_original = reference_image.resize(rotated_image.size)
        diff = ImageChops.difference(rotated_image.convert('L'), resized_original.convert('L'))
        assert diff.getbbox() is not None, f"Rotar {angle} grados no debería resultar en la imagen original reescalada"
    else:
        # Para 0 o 360, debería ser idéntica (o casi idéntica si hay alguna operación mínima)
        diff = ImageChops.difference(rotated_image.convert('L'), reference_image.convert('L'))
        # Permitir una diferencia mínima absoluta si es necesario por artefactos
        extrema_sum = sum(diff.getextrema()) if diff.getbbox() else 0
        assert extrema_sum < 10 , "Rotar 0/360 grados debería resultar en la imagen original"


def test_apply_rotation_invalid_image():
    """Verifica el manejo de entrada None."""
    assert apply_rotation(None, 90) is None

# --- Tests Adicionales (Placeholder - ¡Necesitas añadir los tuyos!) ---
def test_image_processing():
    # Sustituye con pruebas reales para otras funciones como apply_brightness, etc.
    # Ejemplo simple:
    img = Image.new('RGB', (10,10), color=(100,100,100))
    brighter_img = apply_brightness(img, 50) # Factor 50
    assert brighter_img is not None
    # El pixel debe ser mas brillante que (100,100,100)
    assert sum(brighter_img.getpixel((5,5))) > 300
    assert True # Elimina esto cuando añadas aserciones reales