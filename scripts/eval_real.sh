## Real Scenes
# Cat
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/Cat/ --config.scene-name Cat --config.data.white-background False --config.data.view-num-limit 1000 --config.evaluation-only True

# Cat Small
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/CatSmall/ --config.scene-name CatSmall --config.data.white-background False --config.data.view-num-limit 1000 --config.evaluation-only True

# Cup and Fabric
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/CupFabric/ --config.scene-name CupFabric --config.data.white-background False --config.data.view-num-limit 1000 --config.evaluation-only True

# Fish
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/Fish/ --config.scene-name Fish --config.data.white-background False --config.model.geometry-warmup-end 100000 --config.data.view-num-limit 1000 --config.evaluation-only True --config.evaluation-only True

# Furry Scene
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/FurScene/ --config.scene-name FurScene --config.data.white-background False --config.data.view-num-limit 1000 --config.evaluation-only True

# Pikachu
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/Pikachu/ --config.scene-name Pikachu --config.data.white-background False --config.data.view-num-limit 1000 --config.evaluation-only True

# Pixiu
python3 main.py config:nr-hints-cam-opt --config.data.path /path/to/data/Real/Pixiu/ --config.scene-name Pixiu --config.data.white-background False --config.data.view-num-limit 1000 --config.evaluation-only True
