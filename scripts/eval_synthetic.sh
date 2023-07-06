## Synthetic Scenes
# Basket
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Basket_PL_500/ --config.scene-name Basket --config.data.white-background True --config.evaluation-only True

# Layered Woven Ball
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Complex_Ball_PL_500/ --config.scene-name Complex_Ball --config.data.white-background True --config.data.is-z-up True --config.model.sdf-network.init-bias 0.05 --config.evaluation-only True

# Cup Plane Diffuse
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_Diffuse_PL_500/ --config.scene-name Cup_Plane_Diffuse --config.data.white-background True --config.evaluation-only True

# Cup Plane Long Fur
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_LongFur_PL_500/ --config.scene-name Cup_Plane_LongFur --config.data.white-background True --config.evaluation-only True

# Cup Plane Short Fur
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_ShortFur_PL_500/ --config.scene-name Cup_Plane_ShortFur --config.data.white-background True --config.evaluation-only True

# Cup Plane Metal Anisotropic (Anisotropic-Metal)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_Metal_Aniso_PL_500/ --config.scene-name Cup_Plane_Metal_Aniso --config.data.white-background --config.evaluation-only True

# Cup Plane Metal (Metallic)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_Metal_PL_500/ --config.scene-name Cup_Plane_Metal --config.data.white-background True --config.evaluation-only True

# Cup Plane Metal Rough (Glossy-Metal)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_Metal_Rough_PL_500/ --config.scene-name Cup_Plane_Metal_Rough --config.data.white-background True --config.evaluation-only True

# Cup Plane Metal Very Rough (Rough-Metal)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_Metal_VeryRough_PL_500/ --config.scene-name Cup_Plane_Metal_VeryRough --config.data.white-background True --config.evaluation-only True

# Cup Plane Plastic
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_NonMetal_PL_500/ --config.scene-name Cup_Plane_NonMetal --config.data.white-background True --config.evaluation-only True

# Cup Plane Plastic Rough (Glossy-Plastic)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_NonMetal_Rough_PL_500/ --config.scene-name Cup_Plane_NonMetal_Rough --config.data.white-background True --config.evaluation-only True

# Cup Plane Plastic Very Rough (Rough-Plastic)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_NonMetal_VeryRough_PL_500/ --config.scene-name Cup_Plane_NonMetal_VeryRough --config.data.white-background True --config.evaluation-only True

# Cup Plane Sub-Surface Scattering (Translucent)
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Cup_Plane_SSS_PL_500/ --config.scene-name Cup_Plane_SSS --config.data.white-background True --config.evaluation-only True

# Drums
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Drums_PL_500/ --config.scene-name Drums --config.data.white-background True --config.data.is-z-up True --config.evaluation-only True

# Fur Ball
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/FurBall_PL_500/ --config.scene-name FurBall --config.data.white-background True --config.evaluation-only True

# Hotdog
python3 main.py config:nr-hints --config.data.path /path/to/data/Synthetic/Hotdog_PL_500/ --config.scene-name Hotdog --config.data.white-background True --config.data.is-z-up True --config.evaluation-only True

# Lego
python3 main.py config:nr-hints --config.scene-name Lego --config.data.white-background True --config.data.is-z-up True --config.evaluation-only True
