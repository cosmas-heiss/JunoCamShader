import numpy as np

from GUI import GUIObject
from SPICEHandler import SpiceDataHandler
from Renderer import ViewRenderer
from DataDownloader import PerijoveDataDownloader

data_downloader = PerijoveDataDownloader()
spice_data_handler = SpiceDataHandler()
renderer = ViewRenderer(spice_data_handler=spice_data_handler)
gui = GUIObject(renderer=renderer, spice_data_handler=spice_data_handler, data_downloader=data_downloader)
data_downloader.set_gui_object(gui)
renderer.set_gui_object(gui)
gui.mainloop()



'''
TODO:


no height bullshit

do color filters and edge enhancement
deal with the sun when multiple images are loaded / maybe no sun and always fog on the edge, or make selection for sun/no sun
do blending not only based on viewing angle but also on distance
maybe a little bit of sun correction

control reset of newly selected image on unload_image


get rid of unneccessary package dependencies and make code nice

then its kinds finished
'''
