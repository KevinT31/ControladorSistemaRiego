import unittest
from cloud_sync import CloudSync
from unittest.mock import patch

class TestCloudSync(unittest.TestCase):

    @patch('cloud_sync.requests.get')
    def test_verificar_conexion(self, mock_get):
        mock_get.return_value.status_code = 200
        cloud_sync = CloudSync()
        self.assertTrue(cloud_sync.verificar_conexion())

    @patch('cloud_sync.requests.post')
    def test_enviar_datos(self, mock_post):
        cloud_sync = CloudSync()
        sensor_values = {'timestamp': 1234567890, 'humidity': 50.0, 'ph': 7.0, 'ce': 1.5, 'water_level': 80.0}
        cloud_sync.enviar_datos(sensor_values)
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
