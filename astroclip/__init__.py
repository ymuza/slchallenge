# # Re-export modules from subfolders to allow imports directly from astroclip
# from .astroclip import AstroClipModel, ImageHead, SpectrumHead
# from .models import *
# from .astrodino import *
#
#
"""
astroclip package initializer

Este archivo reexporta las clases y funciones más usadas para que puedan
importarse directamente desde `astroclip`, sin necesidad de entrar en subcarpetas.
"""

# Modelos principales (CLIP + encoders)
from .models.astroclip import AstroClipModel, ImageHead, SpectrumHead

# Módulos auxiliares
from .modules import MLP, CrossAttentionHead

# Si quieres exponer también astrodino
from .astrodino import *
