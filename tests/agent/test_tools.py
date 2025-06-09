# En: tests/agent/test_tools.py

import pytest
from PIL import Image, ImageFilter, ImageChops, ImageDraw # <--- IMPORT AÑADIDO
from agent.tools import _apply_filter_impl # Asegurar importación correcta

# --- Fixture de Imagen de Prueba ---
@pytest.fixture(scope="module")
def detailed_image():
    """Imagen con algo de detalle para notar filtros."""
    img = Image.new('RGB', (50, 50), color='white')
    draw = ImageDraw.Draw(img) # <--- Ahora ImageDraw está definido
    draw.line((0, 0, 50, 50), fill='black', width=1)
    draw.line((0, 50, 50, 0), fill='black', width=1)
    return img

# --- Tests Parametrizados para Filtros ---

# Lista de filtros válidos y el objeto PIL correspondiente
VALID_FILTERS = [
    ('blur', ImageFilter.BLUR),
    ('sharpen', ImageFilter.SHARPEN),
    ('smooth', ImageFilter.SMOOTH),
    ('edge_enhance', ImageFilter.EDGE_ENHANCE),
    ('emboss', ImageFilter.EMBOSS),
    ('contour', ImageFilter.CONTOUR),
]

@pytest.mark.parametrize("filter_name, pil_filter", VALID_FILTERS)
def test_apply_filter_impl_valid_filters(detailed_image, filter_name, pil_filter):
    """Verifica la aplicación exitosa de todos los filtros válidos."""
    original_image = detailed_image.copy()
    result_msg, result_img, result_ui = _apply_filter_impl(original_image, filter_name=filter_name)

    # Verificar mensaje de éxito
    assert f"Success: Applied '{filter_name}' filter." in result_msg, f"Mensaje incorrecto para {filter_name}"
    # Verificar que se devuelve una imagen válida
    assert isinstance(result_img, Image.Image), f"Tipo de imagen incorrecto para {filter_name}"
    assert result_img.size == original_image.size, f"Tamaño de imagen incorrecto para {filter_name}"
    # Verificar que no hay actualizaciones de UI
    assert result_ui is None, f"Updates de UI inesperados para {filter_name}"

    # Verificar que la imagen resultante es diferente de la original
    diff = ImageChops.difference(result_img.convert('L'), original_image.convert('L'))
    assert diff.getbbox() is not None, f"El filtro '{filter_name}' no modificó la imagen."

@pytest.mark.parametrize("invalid_filter_name", ["", "nonexistent_filter", "BLUR "])
def test_apply_filter_impl_invalid_filters(detailed_image, invalid_filter_name):
    """Verifica el manejo de nombres de filtro inválidos o desconocidos."""
    result_msg, result_img, result_ui = _apply_filter_impl(detailed_image, filter_name=invalid_filter_name)

    assert "Error: Unknown filter name" in result_msg or "Error:" in result_msg, f"Mensaje de error inesperado para '{invalid_filter_name}'"
    assert result_img is None, "No se debe devolver imagen en caso de error"
    assert result_ui is None, "No debe haber updates de UI en caso de error"
