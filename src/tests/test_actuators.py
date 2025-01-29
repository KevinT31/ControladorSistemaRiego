import unittest
from actuators import PumpControl, ValveControl, FertilizerInjector
from unittest.mock import MagicMock

class TestActuators(unittest.TestCase):

    def setUp(self):
        self.pump = PumpControl(17)
        self.valve_config = {'riego': 27, 'suministro_alternativo': 23}
        self.valve = ValveControl(self.valve_config)
        self.injector = FertilizerInjector(22)

    def test_pump_activation(self):
        self.pump.activar()
        self.assertTrue(self.pump.estado_actual())

    def test_valve_open_close(self):
        self.valve.abrir_valvula('riego')
        self.assertTrue(self.valve.estado_actual('riego'))
        self.valve.cerrar_valvula('riego')
        self.assertFalse(self.valve.estado_actual('riego'))

    def test_injector_activation(self):
        self.injector.activar()
        self.assertTrue(self.injector.estado_actual())

if __name__ == '__main__':
    unittest.main()
