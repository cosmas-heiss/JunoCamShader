import numpy as np
import spiceypy as spice
import json
import os
print(spice.tkvrsn("TOOLKIT"))

class SpiceDataHandler:
    def __init__(self):
        self.clear_kernels()


    def clear_kernels(self):
        spice.kclear()


    def load_kernels(self, data_path, perijove_name):
        kernel_path = os.path.join(data_path, 'global_kernels')
        kernel_list = [os.path.join(kernel_path, x) for x in os.listdir(kernel_path)]
        
        kernel_path = os.path.join(data_path, perijove_name, 'spice_kernels')
        kernel_list += [os.path.join(kernel_path, x) for x in os.listdir(kernel_path)]

        self.clear_kernels()
        spice.furnsh(kernel_list)


    def convert_time_str2et(self, time_str):
        return spice.str2et(time_str)


    def get_sun_direction(self, time_et):
        if type(time_et) is not list:
            time_et = [time_et]
        pos, light_time = spice.spkpos("SUN", time_et, 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
        return np.squeeze(np.array(pos))


    def get_inv_orients_positions(self, times):
        orients, positions = self.get_orients_positions(times)
        inv_orients = [np.linalg.inv(x) for x in orients]
        return inv_orients, positions

    def get_orients_positions(self, times):
        positions, light_time = spice.spkpos("Juno", times, 'IAU_JUPITER', 'NONE', 'JUPITER BARYCENTER')
        positions = [np.array(pos) for pos in positions]

        orients = [np.array(spice.pxform("IAU_JUPITER", "JUNO_JUNOCAM", t)) for t in times]
        return orients, positions