# En: tests/core/test_ai_services.py
import pytest
from unittest.mock import patch, Mock, MagicMock
from PIL import Image
import io
import requests # Necesario importar para mockear errores específicos si es necesario
from core.ai_services import remove_background_ai, _REMBG_AVAILABLE, STABILITY_API_KEY_NAME

@pytest.fixture
def sample_image_ai():
    """Fixture para la imagen de prueba."""
    return Image.new('RGB', (80, 60), color='purple')

@pytest.fixture
def mock_stability_success_response():
    """Fixture para una respuesta simulada exitosa de Stability API."""
    mock_response = Mock(spec=requests.Response) # Usar spec para mejor simulación
    mock_response.status_code = 200
    # Crear bytes de una imagen PNG transparente simulada
    img_byte_arr = io.BytesIO()
    Image.new('RGBA', (80, 60), (0, 0, 0, 0)).save(img_byte_arr, format='PNG')
    mock_response.content = img_byte_arr.getvalue()
    # Simular el método json() en caso de que se llame en manejo de errores (aunque no aquí)
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Mock does not support json", "", 0)
    return mock_response

@pytest.fixture
def mock_stability_failure_response():
    """Fixture para una respuesta simulada de fallo de Stability API."""
    mock_response = Mock(spec=requests.Response)
    mock_response.status_code = 500 # Simular error del servidor
    mock_response.content = b"Internal Server Error"
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Mock does not support json", "", 0)
    mock_response.text = "Internal Server Error"
    return mock_response

@pytest.fixture
def mock_rembg_success_response(sample_image_ai):
    """Fixture para una respuesta simulada exitosa de rembg."""
    # rembg devuelve directamente una imagen PIL
    return Image.new('RGBA', sample_image_ai.size, (255, 0, 0, 128))


# --- Tests usando Context Managers para Patch ---

def test_remove_bg_stability_success(sample_image_ai, mock_stability_success_response):
    """Verifica el flujo exitoso a través de Stability AI."""
    # Usar with para aplicar parches solo durante esta prueba
    with patch('core.ai_services._get_stability_api_key', return_value="fake_key") as mock_get_key, \
         patch('core.ai_services.requests.post', return_value=mock_stability_success_response) as mock_post, \
         patch('core.ai_services.remove_bg_local') as mock_rembg: # Mockear rembg para asegurar que no se llama

        result = remove_background_ai(sample_image_ai)

        mock_get_key.assert_called_once()
        mock_post.assert_called_once()
        # Verificar endpoint y clave en la llamada a post (implícito en _call_stability_api)
        args, kwargs = mock_post.call_args
        assert "remove-background" in args[0] # Verificar parte del endpoint
        assert kwargs['headers']['Authorization'] == "Bearer fake_key"

        mock_rembg.assert_not_called() # No debería haber llamado a rembg
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA' # Stability BG remove devuelve RGBA


def test_remove_bg_fallback_rembg_success(sample_image_ai, mock_stability_failure_response, mock_rembg_success_response):
    """Verifica el fallback a rembg cuando Stability falla."""
    # Solo tiene sentido si rembg está simulado como disponible
    if not _REMBG_AVAILABLE:
        pytest.skip("rembg no está disponible, omitiendo prueba de fallback")

    with patch('core.ai_services._get_stability_api_key', return_value="fake_key") as mock_get_key, \
         patch('core.ai_services.requests.post', return_value=mock_stability_failure_response) as mock_post, \
         patch('core.ai_services.remove_bg_local', return_value=mock_rembg_success_response) as mock_rembg:

        result = remove_background_ai(sample_image_ai)

        mock_get_key.assert_called_once()
        mock_post.assert_called_once() # Se intentó llamar a Stability
        mock_rembg.assert_called_once() # Se llamó a rembg como fallback
        # Verificar que la imagen pasada a rembg fue convertida a RGBA
        rembg_args, _ = mock_rembg.call_args
        assert isinstance(rembg_args[0], Image.Image)
        assert rembg_args[0].mode == 'RGBA'

        assert result == mock_rembg_success_response # El resultado debe ser el de rembg
        assert result.mode == 'RGBA'


def test_remove_bg_no_key_rembg_success(sample_image_ai, mock_rembg_success_response):
    """Verifica el uso de rembg cuando no hay clave API de Stability."""
    if not _REMBG_AVAILABLE:
        pytest.skip("rembg no está disponible, omitiendo prueba sin clave API")

    with patch('core.ai_services._get_stability_api_key', return_value=None) as mock_get_key, \
         patch('core.ai_services.requests.post') as mock_post, \
         patch('core.ai_services.remove_bg_local', return_value=mock_rembg_success_response) as mock_rembg:

        result = remove_background_ai(sample_image_ai)

        mock_get_key.assert_called_once()
        mock_post.assert_not_called() # No se debe llamar a Stability sin clave
        mock_rembg.assert_called_once()
        assert result == mock_rembg_success_response


def test_remove_bg_all_fail(sample_image_ai, mock_stability_failure_response):
    """Verifica el caso donde todo falla."""
    with patch('core.ai_services._get_stability_api_key', return_value="fake_key") as mock_get_key, \
         patch('core.ai_services.requests.post', return_value=mock_stability_failure_response) as mock_post, \
         patch('core.ai_services.remove_bg_local', return_value=None) as mock_rembg: # rembg también falla

        result = remove_background_ai(sample_image_ai)

        mock_get_key.assert_called_once()
        mock_post.assert_called_once() # Intentó Stability
        if _REMBG_AVAILABLE:
            mock_rembg.assert_called_once() # Intentó rembg
        else:
            mock_rembg.assert_not_called() # No intentó rembg si no está disponible

        assert result is None # El resultado final debe ser None